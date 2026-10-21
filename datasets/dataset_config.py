# -*- coding: utf-8 -*-
"""
Dataset registry.

Dict-based registry managing per-dataset root path, channel count and
shape_format.
"""
import os


DATASET_REGISTRY = {
    'PDAC': {
        'root': '/home/wangshengping/DataSet/Neuro',
        'channels': 8,
        'shape_format': 'CHW',
    },
    'NeuroQuantify': {
        'root': '/home/wangshengping/DataSet/SY5Y',
        'channels': 1,
        'shape_format': 'HW',
        'label_suffix': '',
        'normalize': 'zscore',
        'intensity_aug': True,
    },
    'NeuriteOutgrowth': {
        'root': '/home/wangshengping/DataSet/Neurite',
        'channels': 1,
        'shape_format': 'HW',

        'normalize': 'zscore',
        'intensity_aug': True,
        'oversample': True,
    },
    'ARCADE': {
        'root': '/home/wangshengping/DataSet/ARCADE',
        'channels': 1,
        'shape_format': 'HW',
        'label_dir': 'masks',
        'normalize': 'zscore',
        'intensity_aug': True,
        'oversample': True,
        'split_map': {'val': 'train'},
    },
}


def get_dataset_config(name: str) -> dict:
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f'Unknown dataset: {name}, available: {list(DATASET_REGISTRY.keys())}')
    return DATASET_REGISTRY[name]


def get_dataset_paths(name: str, split: str):
    info = get_dataset_config(name)
    root = info['root']
    split = info.get('split_map', {}).get(split, split)
    images_dir = os.path.join(root, split, 'images')
    labels_dir = os.path.join(root, split, info.get('label_dir', 'labels'))
    return images_dir, labels_dir
