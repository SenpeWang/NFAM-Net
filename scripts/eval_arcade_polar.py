# -*- coding: utf-8 -*-
"""4-mode Direction Reversal Evaluation for Native Polar Vector (P / P-Aug) on ARCADE."""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from configs.config import Config
from datasets.dataset import UniversalDataset
from networks.model import build_model
from experiment.representation import install, forbid_q_runtime
from utils.metric import calculate_sample_metrics


def evaluate(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ckpt_path = os.path.join(args.train_dir, 'checkpoints', f'{args.ckpt_type}_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    settings = {
        'sign_grid': [2, 2],
        'sign_probability': 0.5,
        'sign_seed_offset': 1000000,
        'group': args.group,
    }

    # Build model and install polar representation before loading weights
    model = build_model(Config, args.dataset, ablation='full')
    model = install(model, args.group, settings, args.seed)
    forbid_q_runtime()

    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    test_ds = UniversalDataset(args.dataset, 'test', transform=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)

    eval_modes = ['normal', 'global_flip', 'alternate_halfplane', 'quadrants']
    mode_results = {mode: [] for mode in eval_modes}

    print(f'=== Evaluating {args.dataset} {args.group} ({args.ckpt_type}) across 4 reversal modes ===')
    print(f'Test samples count: {len(test_ds)}')

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            image = batch['image'].to(device).float()
            label = batch['label'].numpy()[0]
            case_name = batch['case_name'][0]

            ref_prob = None
            ref_mask = None

            for mode in eval_modes:
                model._axis_eval_mode = mode
                out = model(image)
                prob = F.softmax(out['logits'].float(), dim=1)[0, 1].cpu().numpy()
                pred_mask = prob >= Config.threshold

                metrics = calculate_sample_metrics(prob, label, threshold=Config.threshold)

                if mode == 'normal':
                    ref_prob = prob
                    ref_mask = pred_mask
                    prob_mae = 0.0
                    changed_px = 0
                else:
                    prob_mae = float(np.mean(np.abs(prob - ref_prob)))
                    changed_px = int(np.count_nonzero(pred_mask != ref_mask))

                metrics.update({
                    'case': case_name,
                    'mode': mode,
                    'prob_mae': prob_mae,
                    'changed_pixels': changed_px,
                    'foreground': bool(label.any()),
                })
                mode_results[mode].append(metrics)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(test_loader):
                print(f'Processed [{idx + 1}/{len(test_loader)}] cases...')

    # Reset mode
    model._axis_eval_mode = 'normal'

    summary = {}
    print('\n' + '=' * 80)
    print(f'Summary Report: {args.dataset} | Group: {args.group} | Checkpoint: {args.ckpt_type}')
    print('=' * 80)
    print(f'| Mode | Dice (%) | clDice (%) | Prob MAE | Changed Px |')
    print(f'| :--- | :--- | :--- | :--- | :--- |')

    for mode in eval_modes:
        rows = mode_results[mode]
        mean_dice = float(np.mean([r['dice'] for r in rows])) * 100
        mean_cldice = float(np.mean([r['cldice'] for r in rows])) * 100
        mean_mae = float(np.mean([r['prob_mae'] for r in rows]))
        mean_changed = float(np.mean([r['changed_pixels'] for r in rows]))

        summary[mode] = {
            'dice': mean_dice,
            'cldice': mean_cldice,
            'prob_mae': mean_mae,
            'changed_pixels': mean_changed,
        }
        print(f'| {mode:<20} | {mean_dice:6.2f}% | {mean_cldice:6.2f}% | {mean_mae:8.5f} | {mean_changed:7.1f} |')

    print('=' * 80 + '\n')

    # Save output json
    out_file = os.path.join(args.train_dir, f'eval_{args.ckpt_type}_reversal_results.json')
    with open(out_file, 'w') as f:
        json.dump({'summary': summary, 'per_case': mode_results}, f, indent=2)
    print(f'Evaluation results saved to: {out_file}')


def main():
    parser = argparse.ArgumentParser(description='4-mode Direction Reversal Evaluation for Polar Vector')
    parser.add_argument('--group', type=str, required=True, choices=['P', 'P_AUG', 'Q'])
    parser.add_argument('--dataset', type=str, default='ARCADE')
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--ckpt_type', type=str, default='best_dice', choices=['best_dice', 'best_loss', 'best_ema', 'last'])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
