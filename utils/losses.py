# -*- coding: utf-8 -*-
"""Loss functions for NFAM-Net: Segmentation + Nematic field + Channel Regularization with ablation support."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.qtensor import QTensorTools


class SegLoss(nn.Module):
    """Segmentation loss: Weighted sum of Binary Cross Entropy and Dice loss."""

    def __init__(self, bce_weight=0.3, dice_weight=0.7, eps=1e-5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps
        self.bce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = F.softmax(logits, dim=1)[:, 1]
        targets_f = targets.float()
        intersection = (probs * targets_f).sum(dim=(-2, -1))
        union = probs.sum(dim=(-2, -1)) + targets_f.sum(dim=(-2, -1))
        dice = 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)
        return self.bce_weight * bce + self.dice_weight * dice.mean()


def frank_oseen_loss_2d(q, mask=None):
    """Frank-Oseen-type quadratic elastic energy on 2D Q-tensor."""
    q1 = q[:, 0:1]
    q2 = q[:, 1:2]

    dq1_dx = q1[:, :, :, 1:] - q1[:, :, :, :-1]
    dq2_dx = q2[:, :, :, 1:] - q2[:, :, :, :-1]
    dq1_dy = q1[:, :, 1:, :] - q1[:, :, :-1, :]
    dq2_dy = q2[:, :, 1:, :] - q2[:, :, :-1, :]

    min_h = min(dq1_dx.shape[2], dq1_dy.shape[2])
    min_w = min(dq1_dx.shape[3], dq1_dy.shape[3])

    dq1_dx = dq1_dx[:, :, :min_h, :min_w]
    dq2_dx = dq2_dx[:, :, :min_h, :min_w]
    dq1_dy = dq1_dy[:, :, :min_h, :min_w]
    dq2_dy = dq2_dy[:, :, :min_h, :min_w]

    energy = dq1_dx**2 + dq2_dx**2 + dq1_dy**2 + dq2_dy**2

    if mask is not None:
        mask_2d = mask[:, :, :min_h, :min_w]
        energy = energy * mask_2d
        fg_count = mask_2d.sum().clamp(min=1)
        energy = energy.sum() / fg_count
    else:
        energy = energy.mean()

    return energy


def _compute_mask_and_stats(s, targets):
    """Downsample GT mask to S resolution and compute stats."""
    height, width = s.shape[2], s.shape[3]
    if targets.shape[1] != height or targets.shape[2] != width:
        mask = F.interpolate(targets.float().unsqueeze(1), size=(height, width), mode="nearest")
    else:
        mask = targets.float().unsqueeze(1) if targets.dim() == 3 else targets.float()

    foreground = (mask > 0.5).float()
    background = 1.0 - foreground
    fg_count = foreground.sum().clamp(min=1)
    bg_count = background.sum().clamp(min=1)

    return mask, foreground, background, fg_count, bg_count


def s_supervision_loss(q_pred, targets, s, fg_weight=1.0, bg_weight=1.0):
    """Scalar order parameter supervision loss."""
    if fg_weight <= 0 and bg_weight <= 0:
        return torch.tensor(0.0, device=s.device), {
            "S_foreground_loss": 0.0,
            "S_background_loss": 0.0,
            "S_mean": s.mean().item(),
            "S_foreground_mean": 0.0,
            "S_background_mean": 0.0,
        }

    _, foreground, background, fg_count, bg_count = _compute_mask_and_stats(s, targets)

    loss_foreground = ((1 - s) ** 2 * foreground).sum() / fg_count
    loss_background = (s ** 2 * background).sum() / bg_count
    loss = fg_weight * loss_foreground + bg_weight * loss_background

    components = {
        "S_foreground_loss": loss_foreground.item(),
        "S_background_loss": loss_background.item(),
        "S_mean": s.mean().item(),
        "S_foreground_mean": (s * foreground).sum().div(fg_count).item(),
        "S_background_mean": (s * background).sum().div(bg_count).item(),
    }
    return loss, components


class NematicLoss(nn.Module):
    """Nematic loss: S supervision + Frank-Oseen regularization."""

    def __init__(self, stage_weights=(0.4, 0.3, 0.2, 0.1), fg_weight=1.0, bg_weight=1.0, fo_weight=1.0):
        super().__init__()
        self.stage_weights = list(stage_weights)
        self.fg_weight = fg_weight
        self.bg_weight = bg_weight
        self.fo_weight = fo_weight

    def forward(self, stage_q, targets):
        if not stage_q:
            return torch.tensor(0.0), {}

        device = stage_q[0].device
        total_s_loss = torch.tensor(0.0, device=device)
        total_fo_loss = torch.tensor(0.0, device=device)
        components = {}

        for stage_index, q_pred in enumerate(stage_q):
            stage_weight = self.stage_weights[stage_index] if stage_index < len(self.stage_weights) else 0.0
            if stage_weight <= 0:
                continue

            s = QTensorTools.get_s(q_pred[:, 0:1], q_pred[:, 1:2])
            mask, _, _, _, _ = _compute_mask_and_stats(s, targets)

            s_loss, s_components = s_supervision_loss(
                q_pred, targets, s, self.fg_weight, self.bg_weight)
            total_s_loss = total_s_loss + stage_weight * s_loss

            if self.fo_weight > 0:
                fo_loss = frank_oseen_loss_2d(q_pred, mask)
                total_fo_loss = total_fo_loss + stage_weight * self.fo_weight * fo_loss
                fo_val = (stage_weight * self.fo_weight * fo_loss).item()
            else:
                fo_val = 0.0

            components[f"S/stage{stage_index}/fg_loss"] = s_components["S_foreground_loss"]
            components[f"S/stage{stage_index}/bg_loss"] = s_components["S_background_loss"]
            components[f"S/stage{stage_index}/fg_mean"] = s_components["S_foreground_mean"]
            components[f"S/stage{stage_index}/bg_mean"] = s_components["S_background_mean"]
            components[f"S/stage{stage_index}/mean"] = s_components["S_mean"]
            components[f"FO/stage{stage_index}"] = fo_val

        components["S"] = total_s_loss.item()
        components["FO"] = total_fo_loss.item()
        components["Nematic"] = (total_s_loss + total_fo_loss).item()

        return total_s_loss + total_fo_loss, components


class NFAMNetLoss(nn.Module):
    """NFAM-Net total multi-task loss with robust ablation support."""

    lambda_seg = 1.0
    q_stage_weights = [0.4, 0.3, 0.2, 0.1]

    def __init__(self, lambda_nematic, lambda_cr_info, lambda_cr_b,
                 bce_weight, dice_weight, fo_weight=1.0, s_weight=1.0):
        super().__init__()
        self.lambda_nematic = lambda_nematic
        self.lambda_cr_info = lambda_cr_info
        self.lambda_cr_b = lambda_cr_b

        self.seg_loss = SegLoss(bce_weight, dice_weight)
        fg_w = 1.0 if s_weight > 0 else 0.0
        bg_w = 1.0 if s_weight > 0 else 0.0
        self.s_loss = NematicLoss(stage_weights=self.q_stage_weights, fg_weight=fg_w, bg_weight=bg_w, fo_weight=fo_weight)
        self.loss_components = {}

    def forward(self, outputs, targets):
        seg = self.seg_loss(outputs["logits"], targets)

        stage_q = outputs.get("stage_Q", [])
        if self.lambda_nematic > 0 and len(stage_q) > 0:
            s_loss, s_comps = self.s_loss(stage_q, targets)
        else:
            s_loss = torch.tensor(0.0, device=seg.device)
            s_comps = {}

        total = self.lambda_seg * seg + self.lambda_nematic * s_loss

        cr_losses = outputs.get("cr_losses", {})
        loss_cr_info2 = cr_losses.get("loss_cr_info2", None)
        loss_b2 = cr_losses.get("loss_cr_b2", None)
        loss_cr_info3 = cr_losses.get("loss_cr_info3", None)
        loss_b3 = cr_losses.get("loss_cr_b3", None)

        cr_info_total = torch.tensor(0.0, device=seg.device)
        cr_b_total = torch.tensor(0.0, device=seg.device)

        if loss_cr_info2 is not None and torch.is_tensor(loss_cr_info2):
            cr_info_total = cr_info_total + loss_cr_info2
        if loss_cr_info3 is not None and torch.is_tensor(loss_cr_info3):
            cr_info_total = cr_info_total + loss_cr_info3

        if loss_b2 is not None and torch.is_tensor(loss_b2):
            cr_b_total = cr_b_total + loss_b2
        if loss_b3 is not None and torch.is_tensor(loss_b3):
            cr_b_total = cr_b_total + loss_b3

        total = total + self.lambda_cr_info * cr_info_total + self.lambda_cr_b * cr_b_total

        self.loss_components = {
            "Total": total.item(),
            "Seg": (self.lambda_seg * seg).item(),
            "S_w": (self.lambda_nematic * s_loss).item() if self.lambda_nematic > 0 else 0.0,
            "CR_info": (self.lambda_cr_info * cr_info_total).item() if self.lambda_cr_info > 0 else 0.0,
            "CR_b": (self.lambda_cr_b * cr_b_total).item() if self.lambda_cr_b > 0 else 0.0,
            "R_n2": float(cr_losses.get("R_n2", 0.0)),
            "r_z2": float(cr_losses.get("r_z2", 0.0)),
            "R_n3": float(cr_losses.get("R_n3", 0.0)),
            "r_z3": float(cr_losses.get("r_z3", 0.0)),
            **s_comps,
        }
        return total


def get_loss_function(config):
    """Build total loss function configured from config."""
    return NFAMNetLoss(
        lambda_nematic=config.lambda_nematic,
        lambda_cr_info=config.lambda_cr_info,
        lambda_cr_b=config.lambda_cr_b,
        bce_weight=config.bce_weight,
        dice_weight=config.dice_weight,
        fo_weight=config.fo_weight,
        s_weight=config.s_weight,
    )
