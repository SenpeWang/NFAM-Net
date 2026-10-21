# -*- coding: utf-8 -*-
"""
Visualization utilities.

Includes prediction comparison plots, nematic order field plots and training
curves. Comparison and nematic field figures use a 2x2 layout.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from networks.qtensor import QTensorTools


def plot_predictions(image, gt_mask, pred_mask, save_path,
                     title='', channel_idx=0):
    """
    2x2 segmentation comparison:
      ┌─────────────┬─────────────┐
      │  Input (ch)  │     GT      │
      ├─────────────┼─────────────┤
      │  GT + Pred   │   Pred      │
      │  Overlay     │             │
      └─────────────┴─────────────┘
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    img_show = image[channel_idx] if image.ndim == 3 else image

    axes[0, 0].imshow(img_show, cmap='gray')
    axes[0, 0].set_title('Input', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('Ground Truth', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')

    overlay = np.zeros((*gt_mask.shape, 3), dtype=np.float32)
    gt_bin = (gt_mask > 0).astype(np.float32)
    pred_bin = (pred_mask > 0).astype(np.float32)
    overlay[..., 0] = pred_bin          # R = Pred
    overlay[..., 1] = gt_bin            # G = GT
    overlay[..., 2] = 0
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title('Overlay (G=GT, R=Pred)', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(pred_mask, cmap='gray', vmin=0, vmax=1)
    axes[1, 1].set_title('Prediction', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_nematic_field(s, q1, q2, save_path, gt_mask=None,
                       stride=8, title=''):
    """
    2x2 nematic order field:
      ┌──────────────┬──────────────┐
      │ S heatmap     │ θ angle map   │
      ├──────────────┼──────────────┤
      │ Director arrows│ S + GT contour│
      └──────────────┴──────────────┘
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    h, w = s.shape

    theta, dx, dy = QTensorTools.get_director_numpy(q1, q2)

    im0 = axes[0, 0].imshow(s, cmap='inferno', vmin=0, vmax=1)
    axes[0, 0].set_title('Order Parameter S', fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(theta, cmap='hsv', vmin=-np.pi / 2, vmax=np.pi / 2)
    axes[0, 1].set_title('Director Angle θ', fontsize=11, fontweight='bold')
    axes[0, 1].axis('off')
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    ys = np.arange(stride // 2, h, stride)
    xs = np.arange(stride // 2, w, stride)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
    u = dx[grid_y, grid_x] * s[grid_y, grid_x]
    v = dy[grid_y, grid_x] * s[grid_y, grid_x]

    axes[1, 0].imshow(s, cmap='gray_r', vmin=0, vmax=1, alpha=0.3)
    axes[1, 0].quiver(grid_x, grid_y, u, -v,
                       s[grid_y, grid_x], cmap='coolwarm',
                       scale=25, width=0.003, clim=(0, 1),
                       headwidth=0, headlength=0, headaxislength=0, pivot='middle')
    axes[1, 0].set_title('Nematic Field', fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(s, cmap='inferno', vmin=0, vmax=1, alpha=0.7)
    if gt_mask is not None:
        axes[1, 1].contour(gt_mask, levels=[0.5], colors='lime', linewidths=1.5)
        axes[1, 1].set_title('S + GT Contour', fontsize=11, fontweight='bold')
    else:
        axes[1, 1].set_title('S Field', fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')

    if title:
        fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_loss_curves(history, save_path, title='Training Curves'):
    loss_keys = [k for k in history if 'loss' in k.lower()]
    metric_keys = [k for k in history if k not in loss_keys and len(history[k]) > 0]

    n_panels = sum([bool(loss_keys), bool(metric_keys)])
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0
    if loss_keys:
        ax = axes[panel_idx]
        for k in loss_keys:
            ax.plot(range(1, len(history[k]) + 1), history[k], label=k, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        panel_idx += 1

    if metric_keys:
        ax = axes[panel_idx]
        for k in metric_keys:
            ax.plot(range(1, len(history[k]) + 1), history[k], label=k, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Metrics')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
