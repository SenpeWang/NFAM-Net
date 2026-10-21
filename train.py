# -*- coding: utf-8 -*-
"""NFAM-Net training script with full ablation suite and isolated output support."""

import os
import argparse

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--gpu', type=str, default='0')
_pre_args, _ = _parser.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = _pre_args.gpu
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import warnings
import numpy as np
from collections import OrderedDict
import yaml
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs.config import Config
from datasets.dataset import UniversalDataset, ForegroundOversampleSampler, seed_worker
from networks.model import build_model
from utils.losses import get_loss_function
from utils.stats_logger import DFRStats
from utils.metric import calculate_dice
from utils.visualization import plot_loss_curves

warnings.filterwarnings('ignore')


def cast_outputs_to_float(outputs):
    """Cast network outputs to float32 tensors."""
    out_f32 = {}
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            out_f32[k] = v.float()
        elif isinstance(v, list):
            out_f32[k] = [t.float() if isinstance(t, torch.Tensor) else t for t in v]
        else:
            out_f32[k] = v
    return out_f32


class ModelEMA:
    """Exponential Moving Average for model weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float()
            for k, v in model.state_dict().items() if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow


def save_training_state(path, model, optimizer, scheduler, scaler, epoch, best_dice, best_epoch, best_loss, best_loss_epoch, patience_counter, history):
    """Save full training checkpoint state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_dice': best_dice,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'best_loss_epoch': best_loss_epoch,
        'patience_counter': patience_counter,
        'history': history,
    }, path)


def load_training_state(path, model, optimizer, scheduler, scaler):
    """Load full training checkpoint state."""
    ckpt = torch.load(path, map_location='cuda', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'scaler_state_dict' in ckpt and ckpt['scaler_state_dict'] is not None:
        scaler.load_state_dict(ckpt['scaler_state_dict'])
    return (
        ckpt.get('epoch', 0),
        ckpt.get('best_dice', 0.0),
        ckpt.get('best_epoch', 0),
        ckpt.get('best_loss', float('inf')),
        ckpt.get('best_loss_epoch', 0),
        ckpt.get('patience_counter', 0),
        ckpt.get('history', {'train_loss': [], 'val_loss': [], 'val_dice': []}),
    )


def train_epoch(loader, model, criterion, optimizer, scheduler, scaler, epoch, config, ema=None):
    """Run one epoch of training."""
    model.train()
    losses = []
    loss_meters = {}
    accum = max(int(config.grad_accum), 1)
    dtype = torch.bfloat16 if Config.amp_dtype == 'bfloat16' else torch.float16
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f'Train E{epoch}')
    for step, batch in enumerate(pbar):
        images = batch['image'].cuda(non_blocking=True).float()
        targets = batch['label'].cuda(non_blocking=True).long()
        with autocast(device_type="cuda", enabled=config.amp, dtype=dtype):
            outputs = model(images)
            outputs = cast_outputs_to_float(outputs)
            loss = criterion(outputs, targets)
        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad(set_to_none=True)
            continue
        scaler.scale(loss / accum).backward()
        do_step = ((step + 1) % accum == 0) or ((step + 1) == len(loader))
        if do_step:
            scaler.unscale_(optimizer)
            has_nan = any(
                torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                for p in model.parameters() if p.grad is not None
            )
            if has_nan:
                optimizer.zero_grad(set_to_none=True)
                continue
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
        losses.append(loss.item())
        if step % 50 == 49:
            torch.cuda.empty_cache()
        for k, v in criterion.loss_components.items():
            if isinstance(v, (float, int)):
                loss_meters.setdefault(k, []).append(float(v))
        pbar.set_postfix(OrderedDict(loss=f'{np.mean(losses):.4f}'))
    scheduler.step()
    mean_loss = float(np.mean(losses)) if losses else float('nan')
    mean_comps = {k: float(np.mean(v)) for k, v in loss_meters.items()}
    return mean_loss, mean_comps


@torch.no_grad()
def valid_epoch(loader, model, criterion, epoch, config):
    """Run one epoch of validation."""
    model.eval()
    losses = []
    dice_sum, dice_valid = 0.0, 0
    for batch in tqdm(loader, desc=f'Val E{epoch}'):
        images = batch['image'].cuda(non_blocking=True).float()
        targets = batch['label'].cuda(non_blocking=True).long()
        outputs = cast_outputs_to_float(model(images))
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        d_sum, d_cnt = calculate_dice(outputs["logits"], targets)
        dice_sum += d_sum
        dice_valid += d_cnt
    val_dice = dice_sum / dice_valid if dice_valid > 0 else 0.0
    return float(np.mean(losses)), float(val_dice)


def main():
    """Main training routine."""
    parser = argparse.ArgumentParser(description='NFAM-Net Training')
    parser.add_argument('--dataset', type=str, default='PDAC', choices=['PDAC', 'NeuroQuantify', 'NeuriteOutgrowth', 'ARCADE'])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--seed', type=int, default=1234, help='Random seed (default 1234, same as Config.seed)')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--mode', type=str, default='normal', choices=['normal', 'ex'])
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--train_dir', type=str, default=None, help='Override output run dir (e.g. resume into the original run directory)')
    parser.add_argument('--diag', action='store_true', help='Append _diag to work_dir for diagnostic runs')
    parser.add_argument('--tag', type=str, default='', help='Custom tag appended to work_dir name')
    parser.add_argument('--lambda_nematic', type=float, default=0.3, help='Nematic loss weight (S supervision + FO regularization)')
    parser.add_argument('--lambda_cr_info', type=float, default=0.03, help='CR information preservation loss weight')
    parser.add_argument('--lambda_cr_b', type=float, default=0.05, help='CR bottleneck compression loss weight')
    parser.add_argument('--bce_weight', type=float, default=0.3, help='Segmentation BCE loss weight')
    parser.add_argument('--dice_weight', type=float, default=0.7, help='Segmentation Dice loss weight')
    parser.add_argument('--fo_weight', type=float, default=1.0, help='Frank-Oseen loss weight (0.0 for L2 ablation)')
    parser.add_argument('--s_weight', type=float, default=1.0, help='S order supervision weight (0.0 for L1 ablation)')
    parser.add_argument('--nibs_floor', type=float, default=0.70, help='Soft-NIBS throughput floor (0.0 for hard-gating P1, 0.5 for P2)')
    parser.add_argument('--ema', action=argparse.BooleanOptionalAction, default=True, help='Enable EMA weights (default: on)')
    parser.add_argument('--ablation', type=str, default='full', choices=['full', 'baseline', 'wo_dfe', 'wo_dfr', 'wo_nibs', 'wo_sgate'],
                        help='Ablation study variant mode')
    parser.add_argument('--output_root', type=str, default='./Outputs',
                        help='Root directory for storing outputs (e.g. ./Outputs/Ablation)')
    args = parser.parse_args()

    Config.update_for_dataset(args.dataset)
    Config.seed = args.seed

    # Dynamic loss weighting based on ablation mode
    if args.ablation == 'baseline':
        Config.lambda_nematic = 0.0
        Config.lambda_cr_info = 0.0
        Config.lambda_cr_b = 0.0
    elif args.ablation == 'wo_nibs':
        Config.lambda_nematic = args.lambda_nematic
        Config.lambda_cr_info = 0.0
        Config.lambda_cr_b = 0.0
    else:
        Config.lambda_nematic = args.lambda_nematic
        Config.lambda_cr_info = args.lambda_cr_info
        Config.lambda_cr_b = args.lambda_cr_b

    Config.bce_weight = args.bce_weight
    Config.dice_weight = args.dice_weight
    Config.fo_weight = args.fo_weight
    Config.s_weight = args.s_weight
    Config.nibs_floor = args.nibs_floor

    suffix = '_ex' if args.mode == 'ex' else ''
    if args.diag:
        suffix += '_diag'
    if args.tag:
        suffix += f'_{args.tag}'

    if args.output_root != './Outputs' or args.ablation != 'full' or args.tag:
        ablation_tag = f'_{args.ablation}' if args.ablation != 'full' else '_full'
        Config.work_dir = os.path.join(args.output_root, 'Train', f'{Config.timestamp}_{Config.project_name}_{Config.dataset_name.lower()}{ablation_tag}{suffix}')
    else:
        Config.work_dir = f'./Outputs/Train/{Config.timestamp}_{Config.project_name}_{Config.dataset_name.lower()}{suffix}/'

    if args.train_dir:
        Config.work_dir = os.path.abspath(args.train_dir)

    if args.epochs:
        Config.epochs = args.epochs
        Config.t_max = args.epochs
    if args.batch_size:
        Config.batch_size = args.batch_size
    if args.lr:
        Config.lr = args.lr

    os.makedirs(Config.work_dir, exist_ok=True)
    ckpt_dir = os.path.join(Config.work_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(Config.work_dir, 'config.yaml'), 'w') as f:
        yaml.dump(Config.to_dict(), f, default_flow_style=False, allow_unicode=True)

    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)

    print(f'#---------- Loading Data [{args.dataset}] ----------#')
    if args.mode == 'ex':
        print('[EX Mode] Training on val set, evaluating on test set.')
        train_ds = UniversalDataset(Config.dataset_name, 'val', transform=True)
        val_ds = UniversalDataset(Config.dataset_name, 'test', transform=False)
    else:
        train_ds = UniversalDataset(Config.dataset_name, 'train', transform=True)
        val_ds = UniversalDataset(Config.dataset_name, 'val', transform=False)
    if train_ds.fg_flags is not None:
        sampler = ForegroundOversampleSampler(train_ds.fg_flags, oversample_percent=0.33)
        train_loader = DataLoader(train_ds, batch_size=Config.batch_size, sampler=sampler,
                                  num_workers=Config.num_workers, pin_memory=True, worker_init_fn=seed_worker)
        print(f'[Oversample] foreground oversampling enabled (p=0.33, fg={int(train_ds.fg_flags.sum())}/{len(train_ds)})')
    else:
        train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=Config.num_workers, pin_memory=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False, num_workers=Config.num_workers, pin_memory=True, worker_init_fn=seed_worker)

    tag_str = f' [Tag: {args.tag}]' if args.tag else ''
    print(f'#---------- Building NFAM-Net [Ablation: {args.ablation}{tag_str}] ----------#')
    tb_logger = DFRStats(log_dir=os.path.join(Config.work_dir, 'tensorboard'))
    model = build_model(Config, dataset_name=args.dataset, stats_logger=tb_logger, ablation=args.ablation).cuda().float()
    torch.set_default_dtype(torch.float32)
    for p in model.parameters():
        p.data = p.data.float()
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f'Trainable Params: {trainable_p / 1e6:.2f}M | Total: {total_p / 1e6:.2f}M')

    criterion = get_loss_function(Config).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, betas=Config.betas, eps=Config.eps, weight_decay=Config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.t_max, eta_min=Config.eta_min)
    scaler = GradScaler("cuda", enabled=(Config.amp and Config.amp_dtype == "float16"), init_scale=2.**16 if Config.loss_scale == "dynamic" else 1.0)

    best_dice, best_epoch = 0.0, 0
    best_loss, best_loss_epoch = float('inf'), 0
    best_ema_dice, best_ema_epoch = 0.0, 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_dice': []}
    start_epoch = 1
    ema = ModelEMA(model) if args.ema else None

    if args.resume:
        resume_path = os.path.abspath(args.resume)
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
        if resume_path.endswith('.pt'):
            start_epoch, best_dice, best_epoch, best_loss, best_loss_epoch, patience_counter, history = load_training_state(
                resume_path, model, optimizer, scheduler, scaler,
            )
            start_epoch += 1
            print(f'[RESUME] Full training state from {resume_path}, starting at E{start_epoch}')
        else:
            sd = torch.load(resume_path, map_location='cuda', weights_only=True)
            model.load_state_dict(sd)
            print(f'[RESUME] Model weights only from {resume_path}, starting at E1')

    print('#---------- Training ----------#')
    for epoch in range(start_epoch, Config.epochs + 1):
        tb_logger.set_step(epoch)
        train_loss, train_comps = train_epoch(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, Config, ema=ema)
        for k, v in train_comps.items():
            tb_logger.scalar(f'loss/{k}', v)
        tb_logger.scalar('train/lr', optimizer.param_groups[0].get('lr', Config.lr))
        tb_logger.flush_epoch()
        val_loss, val_dice = valid_epoch(val_loader, model, criterion, epoch, Config)
        torch.cuda.empty_cache()
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        ema_dice = None
        if ema is not None:
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict(ema.state_dict())
            _, ema_dice = valid_epoch(val_loader, model, criterion, epoch, Config)
            torch.cuda.empty_cache()
            model.load_state_dict(backup)
        tag = ' [EX]' if args.mode == 'ex' else ''
        ema_tag = f' | EMA dice={ema_dice:.4f}' if ema_dice is not None else ''
        print(f'E{epoch}{tag} | dice={val_dice:.4f}{ema_tag} | train loss={train_loss:.4f} | val loss={val_loss:.4f}')
        with open(os.path.join(Config.work_dir, 'metrics.txt'), 'a') as f:
            f.write(f'[Epoch {epoch}]{tag} Val Dice={val_dice:.4f}, Val Loss={val_loss:.4f}\n')
            if ema_dice is not None:
                f.write(f'[Epoch {epoch}][EMA] Val Dice={ema_dice:.4f}\n')
        with open(os.path.join(Config.work_dir, 'loss_details.txt'), 'a') as f:
            details = ', '.join([f'{k}={v:.4f}' for k, v in train_comps.items()])
            f.write(f'[Epoch {epoch}]{tag} Train Loss={train_loss:.4f}, {details}\n')
        if args.mode != 'ex':
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'last_model.pth'))
            save_training_state(
                os.path.join(ckpt_dir, 'last_training_state.pt'),
                model, optimizer, scheduler, scaler, epoch,
                best_dice, best_epoch, best_loss, best_loss_epoch, patience_counter, history,
            )

        improved_dice = val_dice > best_dice
        improved_loss = val_loss < best_loss

        if ema is not None and ema_dice is not None and ema_dice > best_ema_dice:
            best_ema_dice, best_ema_epoch = ema_dice, epoch
            if args.mode != 'ex':
                torch.save(ema.state_dict(), os.path.join(ckpt_dir, 'best_ema_model.pth'))
            print(f'>> New Best EMA Dice (Dice={ema_dice:.4f})')

        if improved_dice:
            best_dice, best_epoch = val_dice, epoch
            patience_counter = 0
            if args.mode != 'ex':
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_dice_model.pth'))
            print(f'>> New Best Dice (Dice={val_dice:.4f})')
            with open(os.path.join(Config.work_dir, 'metrics.txt'), 'a') as f:
                f.write(f'  >> New Best Dice at Epoch {epoch}: Dice={val_dice:.4f}\n')
        elif epoch >= Config.early_stopping_start:
            patience_counter += 1

        if improved_loss:
            best_loss, best_loss_epoch = val_loss, epoch
            if args.mode != 'ex':
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_loss_model.pth'))
            print(f' * New Best Loss (Val Loss={val_loss:.4f})')
            with open(os.path.join(Config.work_dir, 'metrics.txt'), 'a') as f:
                f.write(f'  * New Best Val Loss at Epoch {epoch}: Val Loss={val_loss:.4f}\n')
        if args.mode != 'ex' and epoch % Config.val_interval == 0:
            plot_loss_curves(history, os.path.join(Config.work_dir, 'training_curves.png'))
        if Config.early_stopping and epoch >= Config.early_stopping_start and patience_counter >= Config.early_patience:
            print(f'Early stopping at E{epoch}')
            break
    tb_logger.close()
    print(f'Done! Best Dice={best_dice:.4f} at E{best_epoch} | Best Val Loss={best_loss:.4f} at E{best_loss_epoch}')
    if args.mode != 'ex':
        plot_loss_curves(history, os.path.join(Config.work_dir, 'training_curves.png'))
    with open(os.path.join(Config.work_dir, 'summary.json'), 'w') as f:
        json.dump({
            'best_dice': best_dice,
            'best_epoch': best_epoch,
            'best_val_loss': best_loss,
            'best_loss_epoch': best_loss_epoch,
            'dataset': Config.dataset_name,
            'ablation': args.ablation,
            'tag': args.tag,
            'checkpoints': {
                'best_dice': 'checkpoints/best_dice_model.pth',
                'best_loss': 'checkpoints/best_loss_model.pth',
                'last': 'checkpoints/last_model.pth',
            },
        }, f, indent=2)


if __name__ == '__main__':
    main()
