# -*- coding: utf-8 -*-
"""Native Polar Vector (P) and P-Aug training script for ARCADE dataset."""

import os
import sys
import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=str, default='0')
_pre_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = _pre_args.gpu
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from experiment.representation import install, forbid_q_runtime
from experiment.polar_loss import get_loss_function as native_polar_loss
import train

def main():
    parser = argparse.ArgumentParser(description='Train Native Polar Vector on ARCADE')
    parser.add_argument('--group', type=str, required=True, choices=['P', 'P_AUG'],
                        help='Polar field representation: P (native) or P_AUG (sign-augmented)')
    parser.add_argument('--dataset', type=str, default='ARCADE')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output_root', type=str, default='./Outputs/Polar')
    args = parser.parse_args()

    settings = {
        'sign_grid': [2, 2],
        'sign_probability': 0.5,
        'sign_seed_offset': 1000000,
        'group': args.group,
    }

    # Monkey-patch build_model in train module
    orig_build_model = train.build_model

    def polar_builder(*b_args, **b_kwargs):
        model = orig_build_model(*b_args, **b_kwargs)
        model = install(model, args.group, settings, args.seed)
        return model

    train.build_model = polar_builder
    train.get_loss_function = native_polar_loss
    forbid_q_runtime()

    # Configure sys.argv for train.main()
    tag_name = f'arcade_{args.group}_s{args.seed}'
    sys.argv = [
        'train.py',
        '--dataset', args.dataset,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--gpu', str(args.gpu),
        '--seed', str(args.seed),
        '--tag', tag_name,
        '--output_root', args.output_root,
    ]

    print(f'=== Starting Polar Training: Group={args.group}, Dataset={args.dataset}, Seed={args.seed}, GPU={args.gpu} ===')
    train.main()


if __name__ == '__main__':
    main()
