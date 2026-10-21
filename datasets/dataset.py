# -*- coding: utf-8 -*-
"""Dataset loader for NFAM-Net."""

import os
import numpy as np
import tifffile
import torch
from PIL import Image
from torch.utils.data import Dataset
from scipy.ndimage import rotate as ndi_rotate, zoom as ndi_zoom
from .dataset_config import get_dataset_paths, get_dataset_config


AUG_HFLIP_P = 0.5
AUG_VFLIP_P = 0.5
AUG_ROTATE_LIMIT = 35
AUG_ROTATE_P = 0.5
AUG_SCALE_RANGE = (0.7, 1.4)
AUG_SCALE_P = 0.5
AUG_CROP_P = 0.5
DEFAULT_IMG_SIZE = 512
IMAGE_EXTENSIONS = ('.tif', '.tiff', '.png')

IAUG_NOISE_P = 0.1
IAUG_NOISE_VAR = (0.0, 0.1)
IAUG_BLUR_P = 0.2
IAUG_BLUR_SIGMA = (0.5, 1.0)
IAUG_BRIGHTNESS_P = 0.15
IAUG_BRIGHTNESS_RANGE = (0.75, 1.25)
IAUG_CONTRAST_P = 0.15
IAUG_CONTRAST_RANGE = (0.75, 1.25)
IAUG_LOWRES_P = 0.25
IAUG_LOWRES_SCALE = (0.5, 1.0)
IAUG_LOWRES_PER_CHANNEL_P = 0.5
IAUG_GAMMA_P = 0.3
IAUG_GAMMA_INVERT_P = 0.1
IAUG_GAMMA_RANGE = (0.7, 1.5)


def zscore_normalize(image):
    """Apply per-image z-score normalization.

    Args:
        image: (C, H, W) or (H, W) ndarray
    """
    mean = image.mean()
    std = image.std()
    return (image - mean) / (std + 1e-8)


def _sample_bg_contrast(range_lo, range_hi):
    """Sample from the darkening interval with 50% probability, otherwise from
    the brightening interval, following the nnU-Net BGContrast sampling rule."""
    if np.random.random() < 0.5 and range_lo < 1:
        return np.random.uniform(range_lo, 1.0)
    return np.random.uniform(max(range_lo, 1.0), range_hi)


def _gamma_transform_channel(x, gamma, invert):
    """Single-channel gamma transform equivalent to nnU-Net GammaTransform.

    Pipeline: invert if needed, record mean and std, min-max normalize and
    apply the gamma power, rescale back to the original min/max range, then
    restore mean and std, and invert back if needed.
    """
    eps = 1e-7
    if invert:
        x = -x
    mean, std = x.mean(), x.std()
    minm, maxm = x.min(), x.max()
    rnge = maxm - minm
    denom = max(rnge, eps)
    x = ((x - minm) / denom) ** gamma * rnge + minm
    std_here = x.std()
    x = (x - x.mean()) * (std / max(std_here, eps)) + mean
    if invert:
        x = -x
    return x


def apply_intensity_aug(image):
    """Apply nnU-Net default intensity augmentation to the z-scored image.

    Transform order, trigger probabilities, parameter ranges and per-channel
    independence all follow nnUNetTrainer.get_training_transforms.
    """
    from scipy.ndimage import gaussian_filter
    c, h, w = image.shape

    if np.random.rand() < IAUG_NOISE_P:
        variance = np.random.uniform(*IAUG_NOISE_VAR)
        image = image + np.random.normal(0, np.sqrt(variance), image.shape).astype(np.float32)

    if np.random.rand() < IAUG_BLUR_P:
        for ch in range(c):
            if np.random.rand() < 0.5:
                sigma = np.random.uniform(*IAUG_BLUR_SIGMA)
                image[ch] = gaussian_filter(image[ch], sigma)

    if np.random.rand() < IAUG_BRIGHTNESS_P:
        for ch in range(c):
            image[ch] *= _sample_bg_contrast(*IAUG_BRIGHTNESS_RANGE)

    if np.random.rand() < IAUG_CONTRAST_P:
        for ch in range(c):
            factor = _sample_bg_contrast(*IAUG_CONTRAST_RANGE)
            x = image[ch]
            mean, minm, maxm = x.mean(), x.min(), x.max()
            x = (x - mean) * factor + mean
            image[ch] = np.clip(x, minm, maxm)

    if np.random.rand() < IAUG_LOWRES_P:
        for ch in range(c):
            if np.random.rand() < IAUG_LOWRES_PER_CHANNEL_P:
                scale = np.random.uniform(*IAUG_LOWRES_SCALE)
                small = ndi_zoom(image[ch], scale, order=0)
                image[ch] = ndi_zoom(small, (h / small.shape[0], w / small.shape[1]), order=1)

    for apply_prob, invert in ((IAUG_GAMMA_INVERT_P, True), (IAUG_GAMMA_P, False)):
        if np.random.rand() < apply_prob:
            for ch in range(c):
                gamma = _sample_bg_contrast(*IAUG_GAMMA_RANGE)
                image[ch] = _gamma_transform_channel(image[ch], gamma, invert)

    return image.astype(np.float32)


def seed_worker(worker_id):
    """Seed each DataLoader worker independently to avoid duplicated augmentation randomness."""
    import random as _random
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    _random.seed(worker_seed)


class ForegroundOversampleSampler(torch.utils.data.Sampler):
    """Foreground oversampling sampler.

    Each epoch draws oversample_percent of samples from foreground-containing
    images and the rest from the full sample pool.
    """

    def __init__(self, fg_flags, oversample_percent=0.33):
        self.fg_idx = np.where(np.asarray(fg_flags))[0]
        self.all_idx = np.arange(len(fg_flags))
        self.oversample_percent = oversample_percent
        self.num_samples = len(fg_flags)

    def __iter__(self):
        n_fg = int(round(self.num_samples * self.oversample_percent))
        picks = np.concatenate([
            np.random.choice(self.fg_idx, n_fg, replace=True),
            np.random.choice(self.all_idx, self.num_samples - n_fg, replace=True),
        ])
        np.random.shuffle(picks)
        return iter(picks.tolist())

    def __len__(self):
        return self.num_samples


def _load_image(path, shape_format):
    if path.lower().endswith(('.tif', '.tiff')):
        image = tifffile.imread(path)
    else:
        image = np.array(Image.open(path))

    if shape_format == 'HWC':
        image = np.transpose(image, (2, 0, 1))
    elif shape_format == 'HW':
        if image.ndim == 3:
            image = np.max(image, axis=2)
        image = image[np.newaxis, :]

    image = image.astype(np.float32)
    max_value = image.max() if image.size > 0 else 0.0
    image = image / 65535.0 if max_value > 255 else image / 255.0
    return image.astype(np.float32)


def _load_label(label_path):
    label = tifffile.imread(label_path) if label_path.endswith(('.tif', '.tiff')) else np.array(Image.open(label_path))
    if label.ndim == 3:
        label = np.max(label, axis=2)
    return (label > 0).astype(np.int64)


def _resize_spatial(image, label, target_size):
    height, width = image.shape[-2], image.shape[-1]
    if (height, width) == (target_size, target_size):
        return image, label
    zoom_h = target_size / height
    zoom_w = target_size / width
    image = ndi_zoom(image, (1, zoom_h, zoom_w), order=1)
    label = ndi_zoom(label.astype(np.float32), (zoom_h, zoom_w), order=0).astype(np.int64)
    return image, label


def apply_train_aug(image, label):
    if np.random.rand() < AUG_SCALE_P:
        scale = np.random.uniform(AUG_SCALE_RANGE[0], AUG_SCALE_RANGE[1])
        h, w = image.shape[1], image.shape[2]

        image = ndi_zoom(image, (1, scale, scale), order=1)
        label = ndi_zoom(label, (scale, scale), order=0).astype(np.int64)

        c, h_new, w_new = image.shape
        if h_new != h or w_new != w:
            h_crop = min(h_new, h)
            w_crop = min(w_new, w)
            image = image[:, :h_crop, :w_crop]
            label = label[:h_crop, :w_crop]

            c, h_cur, w_cur = image.shape
            if h_cur < h or w_cur < w:
                pad_h = h - h_cur
                pad_w = w - w_cur
                image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
                label = np.pad(label, ((0, pad_h), (0, pad_w)), mode='constant')

    if np.random.rand() < AUG_CROP_P:
        h, w = image.shape[1], image.shape[2]
        crop_size = min(h, w, 512)

        if h > crop_size and w > crop_size:
            top = np.random.randint(0, h - crop_size)
            left = np.random.randint(0, w - crop_size)

            image = image[:, top:top+crop_size, left:left+crop_size]
            label = label[top:top+crop_size, left:left+crop_size]

    if np.random.rand() < AUG_HFLIP_P:
        image = image[:, :, ::-1]
        label = label[:, ::-1]

    if np.random.rand() < AUG_VFLIP_P:
        image = image[:, ::-1, :]
        label = label[::-1, :]

    if np.random.rand() < AUG_ROTATE_P and AUG_ROTATE_LIMIT > 0:
        angle = float(np.random.uniform(-AUG_ROTATE_LIMIT, AUG_ROTATE_LIMIT))
        image = ndi_rotate(image, angle, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0.0)
        label = ndi_rotate(label, angle, reshape=False, order=0, mode='constant', cval=0).astype(np.int64)

    return image, label


class UniversalDataset(Dataset):
    def __init__(self, dataset_name, split='train', transform=True, img_size=None):
        self.dataset_name = dataset_name
        self.split = split
        self.img_size = img_size if img_size is not None else DEFAULT_IMG_SIZE
        self.do_augment = transform and (split == 'train')

        self.images_dir, self.labels_dir = get_dataset_paths(dataset_name, split)
        ds_info = get_dataset_config(dataset_name)
        self.channels = ds_info['channels']
        self.shape_format = ds_info['shape_format']
        self.label_suffix = ds_info.get('label_suffix', '')
        self.use_zscore = ds_info.get('normalize') == 'zscore'
        self.use_intensity_aug = ds_info.get('intensity_aug', False) and self.use_zscore
        self.use_oversample = ds_info.get('oversample', False) and (split == 'train')
        self.fg_flags = None

        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ])
        if self.use_oversample:
            self.fg_flags = self._compute_fg_flags()

        print(
            f"[DatasetInfo] {self.dataset_name}/{split}: {self.images_dir} "
            f"({len(self.image_files)} samples, ch={self.channels}, "
            f"aug={self.do_augment}, zscore={self.use_zscore}, "
            f"iaug={self.use_intensity_aug and self.do_augment}, oversample={self.use_oversample})"
        )

    def _compute_fg_flags(self):
        """Precompute per-image foreground flags of the training set for the oversampler."""
        flags = []
        for f in self.image_files:
            label = _load_label(self._label_path(f))
            flags.append(label.sum() > 0)
        return np.asarray(flags)

    def __len__(self):
        return len(self.image_files)

    def _label_path(self, img_name):
        base = os.path.splitext(img_name)[0]
        if self.label_suffix:
            label_name = base + self.label_suffix + '.png'
        else:
            label_name = base + '.png'
        path = os.path.join(self.labels_dir, label_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing label for {img_name}: expected {path}')
        return path

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        case_name = img_name.rsplit('.', 1)[0]
        image = _load_image(os.path.join(self.images_dir, img_name), self.shape_format)
        if self.use_zscore:
            image = zscore_normalize(image)
        label = _load_label(self._label_path(img_name))

        image, label = _resize_spatial(image, label, self.img_size)

        if self.do_augment:
            image, label = apply_train_aug(image, label)
            if self.use_intensity_aug:
                image = apply_intensity_aug(image)

        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        label = torch.from_numpy(np.ascontiguousarray(label)).long()

        if torch.isnan(image).any() or torch.isinf(image).any():
            image = torch.zeros_like(image)

        return {
            'image': image,
            'label': label,
            'case_name': case_name,
        }
