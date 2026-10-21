# -*- coding: utf-8 -*-
"""NOFM: Nematic-guided Order-Field Modulated Block for deep stages (Stage 2 & 3).

Replaces standard FFN inside VSSBlock with a Nematic-Guided Gated Linear Unit (N-GLU):
1. SS2D: 4-directional selective scan for global long-range context.
2. Depthwise 3x3 Conv: captures local 2D director field continuity.
3. SwiGLU Gating: content-dependent non-linear routing.
4. Nematic Order Gating: modulates gate throughput using local scalar order parameter S.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .VMamba.vmamba import SS2D, LayerNorm, DropPath
from .qtensor import QTensorTools


class NematicGLU(nn.Module):
    """Nematic Gated Linear Unit (N-GLU).

    Args:
        dim: Feature channel dimension C.
        hidden_features: Hidden dimension for SwiGLU. Defaults to int(dim * 8 / 3).
        drop: Dropout rate. Defaults to 0.0.
    """

    def __init__(self, dim: int, hidden_features: int = None, drop: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or int(dim * 8 / 3)
        hidden_features = (hidden_features // 8) * 8
        self.dim = dim
        self.hidden_features = hidden_features

        # Projection to gate and value branches
        self.w_gate = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=False)
        self.w_val = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=False)

        # 2D local depthwise continuity
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features, bias=False
        )
        self.dw_norm = nn.GroupNorm(8, hidden_features)

        # Nematic scalar order parameter S modulation (channel-wise projection)
        self.s_proj = nn.Conv2d(1, hidden_features, kernel_size=1, bias=True)
        nn.init.zeros_(self.s_proj.weight)
        nn.init.zeros_(self.s_proj.bias)

        # Output projection back to C
        self.w_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor, s: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through Nematic-GLU.

        Args:
            x: Input feature tensor [B, C, H, W].
            s: Nematic scalar order parameter tensor [B, 1, H, W] or None.

        Returns:
            Output feature tensor [B, C, H, W].
        """
        gate = self.w_gate(x)
        if s is not None:
            if s.shape[-2:] != x.shape[-2:]:
                s = F.interpolate(s, size=x.shape[-2:], mode="bilinear", align_corners=False)
            gate = gate + self.s_proj(s)

        gate_act = F.silu(gate)
        val = self.w_val(x)
        val_local = self.dw_norm(self.dwconv(val))

        out = gate_act * val_local
        out = self.w_out(out)
        return self.drop(out)


class NOFMBlock(nn.Module):
    """Nematic-Guided Gated Linear Block for deep stages (Stage 2 & 3).

    Args:
        dim: Feature channel dimension.
        drop_path: Stochastic depth rate. Defaults to 0.0.
        chunk_size: Scan chunk size for SS2D. Defaults to 64.
        mlp_ratio: Ratio for GLU hidden dimension. Defaults to 4.0.
        enable_sgate: Whether to inject scalar order parameter S into N-GLU gate.
            Defaults to True. When False, S=None is passed (ablation: w/o S-Gate).
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        chunk_size: int = 64,
        mlp_ratio: float = 4.0,
        enable_sgate: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.norm_ssm = LayerNorm(dim, channel_first=True)
        self.ssm = SS2D(dim, chunk_size=chunk_size)
        self.drop_path = DropPath(drop_path)

        self.norm_glu = LayerNorm(dim, channel_first=True)
        hidden_dim = int(dim * mlp_ratio * 2 / 3)
        self.nglu = NematicGLU(dim, hidden_features=hidden_dim)
        self.enable_sgate = enable_sgate

    def forward(self, x: torch.Tensor, q: torch.Tensor = None, *args, **kwargs) -> torch.Tensor:
        """Forward pass through NOFMBlock.

        Args:
            x: Input feature tensor [B, C, H, W].
            q: Stage Q-tensor [B, 2, H, W] or None.

        Returns:
            Modulated feature tensor [B, C, H, W].
        """
        # Branch 1: 4-directional 2D selective scan (global context)
        x = x + self.drop_path(self.ssm(self.norm_ssm(x)))

        # Branch 2: Nematic-Gated Linear Unit (local 2D continuity + nematic order gating)
        if self.enable_sgate and q is not None and q.shape[1] >= 2:
            s = QTensorTools.get_s(q[:, 0:1], q[:, 1:2])
        else:
            s = None

        x = x + self.drop_path(self.nglu(self.norm_glu(x), s))
        return x
