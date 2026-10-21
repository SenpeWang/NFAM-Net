# -*- coding: utf-8 -*-
"""NFAM-Net testing script with ablation support, fast metrics-only mode, and isolated output."""

import os
import re
import json
import argparse
import warnings
from datetime import datetime

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=str, default='0')
_pre_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = _pre_args.gpu
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

from configs.config import Config
from datasets.dataset import UniversalDataset
from networks.model import build_model
from networks.qtensor import QTensorTools
from utils.metric import calculate_sample_metrics
from utils.visualization import plot_predictions, plot_nematic_field

warnings.filterwarnings('ignore')


CKPT_FILES = {
    'best_dice': 'best_dice_model.pth',
    'best_loss': 'best_loss_model.pth',
    'best_ema': 'best_ema_model.pth',
    'last': 'last_model.pth',
}
CKPT_TYPES_BY_FILE = {v: k for k, v in CKPT_FILES.items()}


def infer_ckpt_type_from_path(path, default='best_dice'):
    """Infer checkpoint type from filename."""
    return CKPT_TYPES_BY_FILE.get(os.path.basename(path), default)


def resolve_checkpoint(train_dir=None, ckpt_type='best_dice', ckpt=None):
    """Locate model checkpoint file from arguments."""
    if ckpt:
        ckpt_path = os.path.abspath(ckpt)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')
        train_dir = os.path.dirname(os.path.dirname(ckpt_path))
        ckpt_type = infer_ckpt_type_from_path(ckpt_path, ckpt_type)
        return ckpt_path, train_dir, ckpt_type
    if not train_dir:
        raise ValueError('Either --ckpt or --train_dir is required')
    if ckpt_type not in CKPT_FILES:
        raise ValueError(f'Unknown ckpt_type={ckpt_type}; choose one of {sorted(CKPT_FILES)}')
    ckpt_path = os.path.join(train_dir, 'checkpoints', CKPT_FILES[ckpt_type])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')
    return ckpt_path, train_dir, ckpt_type


def detect_ablation_mode(train_dir, default='full'):
    """Detect ablation variant from train_dir or its summary.json."""
    if not train_dir:
        return default
    summary_path = os.path.join(train_dir, 'summary.json')
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r') as f:
                data = json.load(f)
                if 'ablation' in data:
                    return data['ablation']
        except Exception:
            pass

    folder = os.path.basename(os.path.normpath(train_dir)).lower()
    for mode in ['baseline', 'wo_dfe', 'wo_dfr', 'wo_nibs', 'wo_sgate']:
        if mode in folder:
            return mode
    return default


def main():
    """Main testing routine."""
    parser = argparse.ArgumentParser(description='NFAM-Net Testing')
    parser.add_argument('--dataset', type=str, required=True, choices=['PDAC', 'NeuroQuantify', 'NeuriteOutgrowth', 'ARCADE'])
    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--ckpt', dest='ckpt', type=str, default=None)
    parser.add_argument('--ckpt_type', type=str, default='best_dice', choices=['best_dice', 'best_loss', 'best_ema', 'last'])
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--vis', action='store_true', help='Save comparison (4-grid) + nematic field in addition to pred')
    parser.add_argument('--ablation', type=str, default=None, choices=['full', 'baseline', 'wo_dfe', 'wo_dfr', 'wo_nibs', 'wo_sgate'])
    parser.add_argument('--output_root', type=str, default=None, help='Custom output root directory')
    parser.add_argument('--no_save_images', action='store_true', help='Skip saving PNG masks; evaluate metrics only')
    args = parser.parse_args()

    ckpt_path, train_dir, ckpt_type = resolve_checkpoint(args.train_dir, args.ckpt_type, args.ckpt)
    dataset_name = args.dataset

    ablation_mode = args.ablation or detect_ablation_mode(train_dir, default='full')

    # Resolve output directory
    train_folder = os.path.basename(os.path.normpath(train_dir))
    if args.output_root:
        Config.test_dir = os.path.join(args.output_root, 'Test', f'{train_folder}_{ckpt_type}')
    elif 'Ablation' in train_dir:
        out_root = os.path.dirname(os.path.dirname(os.path.normpath(train_dir)))
        Config.test_dir = os.path.join(out_root, 'Test', f'{train_folder}_{ckpt_type}')
    else:
        train_ts_match = re.search(r'(\d{8}_\d{6}(?:_\d{3})?)', ckpt_path)
        train_ts = train_ts_match.group(1) if train_ts_match else Config.timestamp
        Config.test_dir = f'./Outputs/Test/{train_ts}_{Config.project_name}_{dataset_name.lower()}_{ckpt_type}/'

    os.makedirs(Config.test_dir, exist_ok=True)

    print(f'#---------- Loading Test Dataset [{dataset_name}] ----------#')
    test_ds = UniversalDataset(dataset_name, 'test', transform=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    print(f'#---------- Building Model [Ablation: {ablation_mode}] ----------#')
    model = build_model(Config, dataset_name=dataset_name, ablation=ablation_mode).cuda()
    sd = torch.load(ckpt_path, map_location='cuda', weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[Checkpoint Info] strict=False loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval()
    print(f'Loaded weights: {ckpt_path}')

    pred_dir = os.path.join(Config.test_dir, 'pred')
    if not args.no_save_images:
        os.makedirs(pred_dir, exist_ok=True)

    comparison_dir = None
    stage_dirs = None
    if args.vis:
        comparison_dir = os.path.join(Config.test_dir, 'comparison')
        nematic_dir = os.path.join(Config.test_dir, 'nematic')
        os.makedirs(comparison_dir, exist_ok=True)
        os.makedirs(nematic_dir, exist_ok=True)
        stage_dirs = []
        for i in range(4):
            stage_dir = os.path.join(nematic_dir, f'stage{i}')
            os.makedirs(stage_dir, exist_ok=True)
            stage_dirs.append(stage_dir)

    all_metrics = []
    dtype = torch.bfloat16 if Config.amp_dtype == 'bfloat16' else torch.float16

    if args.no_save_images:
        mode_str = 'METRICS ONLY (no images saved)'
    elif args.vis:
        mode_str = 'ALL (pred + comparison + nematic)'
    else:
        mode_str = 'pred masks only'
    print(f'#---------- Testing ({mode_str}) ----------#')

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Test [{ablation_mode}]'):
            images = batch['image'].cuda().float()
            targets = batch['label'].numpy()[0]
            case_name = batch['case_name'][0]
            with autocast(device_type='cuda', enabled=Config.amp, dtype=dtype):
                outputs = model(images)
            pred_prob = F.softmax(outputs['logits'].float(), dim=1)[0, 1].cpu().numpy()
            pred_mask = (pred_prob >= Config.threshold).astype(np.uint8)
            metrics = calculate_sample_metrics(pred_prob, targets, threshold=Config.threshold)
            metrics['case'] = case_name
            all_metrics.append(metrics)

            if not args.no_save_images:
                pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8))
                pred_mask_img.save(os.path.join(pred_dir, f'{case_name}.png'))

            if args.vis:
                plot_predictions(
                    batch['image'][0].numpy(), targets, pred_mask,
                    os.path.join(comparison_dir, f'{case_name}.png'),
                    title=case_name,
                )

                stage_q = outputs.get('stage_Q', [])
                for stage_idx, q in enumerate(stage_q):
                    q_np = q[0].cpu().numpy()
                    q1 = q_np[0]
                    q2 = q_np[1]
                    s = QTensorTools.get_s(
                        torch.tensor(q1).unsqueeze(0).unsqueeze(0),
                        torch.tensor(q2).unsqueeze(0).unsqueeze(0)
                    ).numpy()[0, 0]

                    h_s, w_s = s.shape
                    if targets.shape[0] != h_s or targets.shape[1] != w_s:
                        gt_downsampled = F.interpolate(
                            torch.tensor(targets).float().unsqueeze(0).unsqueeze(0),
                            size=(h_s, w_s), mode='nearest'
                        ).numpy()[0, 0]
                    else:
                        gt_downsampled = targets

                    plot_nematic_field(
                        s, q1, q2,
                        os.path.join(stage_dirs[stage_idx], f'{case_name}.png'),
                        gt_mask=gt_downsampled,
                        title=f'{case_name} Stage{stage_idx}',
                    )

    valid_metrics = [m for m in all_metrics if m.get('valid', True)]
    skipped = len(all_metrics) - len(valid_metrics)
    print('\n' + '=' * 60)
    metric_names = ['dice', 'iou', 'cldice', 'precision', 'recall']
    summary = {}
    for name in metric_names:
        vals = [m[name] for m in valid_metrics if np.isfinite(m[name])]
        mean_v = np.mean(vals) if vals else float('nan')
        std_v = np.std(vals) if vals else float('nan')
        summary[name] = {'mean': float(mean_v), 'std': float(std_v)}
        print(f'  {name:>10s}: {mean_v:.4f} +/- {std_v:.4f}')
    print(f'  Valid: {len(valid_metrics)} / Total: {len(all_metrics)} (skipped {skipped} empty)')
    print('=' * 60)

    report = {
        'dataset': dataset_name,
        'checkpoint': ckpt_path,
        'ckpt_type': ckpt_type,
        'ablation': ablation_mode,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'num_samples': len(all_metrics),
        'summary': summary,
        'per_sample': all_metrics,
    }
    with open(os.path.join(Config.test_dir, 'test_report.json'), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(os.path.join(Config.test_dir, 'test_report.txt'), 'w') as f:
        f.write(f'NFAM-Net Test | {dataset_name} | Ablation: {ablation_mode} | {ckpt_type} | {len(all_metrics)} samples\n\n')
        for name in metric_names:
            s = summary[name]
            f.write(f'{name:>10s}: {s["mean"]:.4f} +/- {s["std"]:.4f}\n')

    print(f'Saved test report: {Config.test_dir}')


if __name__ == '__main__':
    main()
