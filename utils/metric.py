# -*- coding: utf-8 -*-
"""
Evaluation metrics for binary foreground/background segmentation.

Includes Dice, IoU, clDice, Precision and Recall.
All metrics are macro-averaged with equal weight per sample.
"""

import numpy as np
import torch
from skimage.morphology import skeletonize


def calculate_dice(pred, target):
    """
    Batch-level binary Dice with macro averaging.

    Foreground Dice is computed independently per sample and averaged
    arithmetically over samples. Samples where both GT and prediction
    are all zeros are treated as invalid and excluded from the mean.

    Args:
        pred:   (B, C, H, W) logits or (B, H, W) class indices
        target: (B, H, W) class indices {0, 1}
    Returns:
        dice_sum and valid_count: the accumulated foreground Dice and
        the number of valid samples
    """
    if pred.dim() == 4:
        pred = torch.argmax(pred, dim=1)

    batch_size = pred.shape[0]
    dice_sum, valid = 0.0, 0

    for i in range(batch_size):
        p = (pred[i] > 0).float()
        t = (target[i] > 0).float()

        if p.sum() == 0 and t.sum() == 0:
            continue

        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dice_sum += (2 * inter / union).item()
        valid += 1

    return dice_sum, valid


def cl_dice(pred_bin, gt_bin):
    """
    Centerline Dice: skeleton-based topological connectivity metric
    measuring the continuity of elongated neurite structures.

    T_prec = |pred ∩ skel(gt)| / |skel(gt)|
    T_sens = |gt ∩ skel(pred)| / |skel(pred)|
    clDice = 2·T_prec·T_sens / (T_prec + T_sens)
    """
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 1.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return 0.0

    skel_pred = skeletonize(pred_bin.astype(bool))
    skel_gt = skeletonize(gt_bin.astype(bool))

    if skel_pred.sum() == 0 or skel_gt.sum() == 0:
        return 0.0

    t_prec = np.sum(pred_bin * skel_gt) / np.sum(skel_gt)
    t_sens = np.sum(gt_bin * skel_pred) / np.sum(skel_pred)
    return 2 * t_prec * t_sens / (t_prec + t_sens + 1e-8)


def calculate_sample_metrics(pred, gt, threshold=0.5):
    """
    Compute the full metric set for a single binary sample.

    Samples where both GT and prediction are all zeros are marked
    invalid (valid=False) and excluded from the mean.

    Args:
        pred:      (H, W) float prediction probabilities
        gt:        (H, W) {0, 1} labels
        threshold: binarization threshold
    Returns:
        dict: dice, iou, precision, recall, cldice, valid
    """
    pred_bin = (pred >= threshold).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)

    if gt_bin.sum() == 0 and pred_bin.sum() == 0:
        return {
            'dice': 0.0, 'iou': 0.0,
            'precision': 0.0, 'recall': 0.0,
            'cldice': 0.0,
            'valid': False,
        }

    inter = np.sum(pred_bin * gt_bin)
    pred_sum = np.sum(pred_bin)
    gt_sum = np.sum(gt_bin)
    union = pred_sum + gt_sum - inter

    tp = inter
    fp = pred_sum - inter
    fn = gt_sum - inter

    return {
        'dice': (2 * inter) / (pred_sum + gt_sum + 1e-8),
        'iou': inter / (union + 1e-8),
        'precision': tp / (tp + fp + 1e-8),
        'recall': tp / (tp + fn + 1e-8),
        'cldice': cl_dice(pred_bin, gt_bin),
        'valid': True,
    }
