# -*- coding: utf-8 -*-
"""Soft-Bounded Nematic Information Bottleneck Selection (Soft-NIBS) module."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NematicInformationBottleneckSelection(nn.Module):
    """Soft-Bounded Nematic Information Bottleneck Selection (Soft-NIBS).

    Deep feature channel selector with a protective throughput floor, ensuring
    no channel is completely eliminated while selectively boosting nematic-rich channels.

    Forward Pass:
        1. Channel activation statistics:
           a_c = GAP(|F_{d, c}|)
        2. Channel selection gate:
           z = Sigmoid(W_2 GELU(W_1 a)) in (0, 1)
        3. Soft-bounded channel modulation:
           w = floor + (1.0 - floor) * z in [floor, 1.0]
        4. Feature selection:
           F_prime = w * F_d

    Training Objective:
        1. Nematic information preservation loss:
           R_n = sum(A_prime * S_d) / (sum(A_prime) + eps)
           L_cr_info = - R_n
        2. Information Bottleneck constraint:
           r_z = mean(z)
           L_cr_b = (r_z - rho)^2

    Args:
        dim: Feature channel dimension C.
        reduction: Channel reduction ratio in selector MLP. Defaults to 4.
        floor: Minimum channel throughput floor. Defaults to 0.70.
        rho: Target channel retention ratio. Defaults to 0.85.
        init_bias: Initial bias for final linear layer. Defaults to 1.5.
        eps: Numerical stability epsilon. Defaults to 1e-6.
        enable_nibs: Whether to apply soft channel selection. Defaults to True.
            When False, acts as identity pass with empty loss_dict (ablation: w/o Soft-NIBS).
    """

    def __init__(
        self,
        dim: int,
        reduction: int = 4,
        floor: float = 0.70,
        rho: float = 0.85,
        init_bias: float = 1.5,
        eps: float = 1e-6,
        enable_nibs: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.reduction = reduction
        self.floor = floor
        self.rho = rho
        self.eps = eps
        self.enable_nibs = enable_nibs

        hidden_dim = max(dim // reduction, 16)
        self.selector = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=True),
            nn.Sigmoid(),
        )

        nn.init.normal_(self.selector[2].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.selector[2].bias, init_bias)

    def forward(self, x: torch.Tensor, s_raw: torch.Tensor = None):
        """Forward selection and optional training loss computation.

        Args:
            x: Deep feature tensor of shape [B, C, H, W].
            s_raw: Raw scalar order parameter of shape [B, 1, H_in, W_in].
                Defaults to None.

        Returns:
            Tuple[torch.Tensor, dict]: (Selected feature F_prime, loss_dict).
        """
        if not self.enable_nibs:
            # Ablation: w/o Soft-NIBS (identity pass, no channel modulation, no CR loss)
            return x, {}

        b, c, h, w = x.shape

        # Step 1: Channel activation statistics a_c = GAP(|F_{d, c}|)
        a = torch.abs(x).mean(dim=(-2, -1))

        # Step 2: Channel selection gate z in (0, 1)^C
        z = self.selector(a)

        # Step 3: Soft-bounded throughput modulation w in [floor, 1.0]
        w_mod = self.floor + (1.0 - self.floor) * z

        # Step 4: Feature modulation
        x_prime = w_mod.view(b, c, 1, 1) * x

        loss_dict = {}
        if self.training and s_raw is not None:
            if s_raw.shape[-2:] != (h, w):
                s_d = F.adaptive_avg_pool2d(s_raw, (h, w)).clamp(0.0, 1.0)
            else:
                s_d = s_raw.clamp(0.0, 1.0)

            a_prime = torch.abs(x_prime).float()
            s_d_f = s_d.float()

            numerator = (a_prime * s_d_f).sum(dim=(1, 2, 3))
            denominator = a_prime.sum(dim=(1, 2, 3)) + self.eps
            r_n = numerator / denominator

            loss_nem = - r_n.mean()

            r_z = z.float().mean(dim=-1)
            loss_bottleneck = ((r_z - self.rho) ** 2).mean()

            loss_dict = {
                "loss_cr_info": loss_nem,
                "loss_cr_b": loss_bottleneck,
                "R_n": r_n.mean().detach(),
                "r_z": r_z.mean().detach(),
                "r_w": w_mod.float().mean().detach(),
            }

        return x_prime, loss_dict
