"""Native vector amplitude supervision and signed polar smoothness."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.losses import SegLoss
from experiment.representation import amplitude


def polar_edge_energy(a, b):
    """Match Q's infinitesimal angular stiffness without constructing Q.

    (r_i-r_j)^2 + 8*(r_i*r_j - p_i dot p_j).
    For equal amplitude this is 8*r^2*(1-cos(delta theta)); near zero angle
    it equals 4*r^2*delta theta^2. It penalizes opposite signed vectors.
    Radial stiffness is one, as in the source Q component smoothness.
    """
    ra = torch.linalg.vector_norm(a, dim=1, keepdim=True)
    rb = torch.linalg.vector_norm(b, dim=1, keepdim=True)
    dot = (a * b).sum(dim=1, keepdim=True)
    return (ra - rb).square() + 8 * (ra * rb - dot).clamp_min(0)


def polar_smoothness(p, mask):
    h, w = p.shape[-2:]
    a = p[:, :, :h - 1, :w - 1]
    e = polar_edge_energy(a, p[:, :, 1:, :w - 1]) + polar_edge_energy(a, p[:, :, :h - 1, 1:])
    m = mask[:, :, :h - 1, :w - 1]
    return (e * m).sum() / m.sum().clamp_min(1)


class NativePolarLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.seg_loss = SegLoss(.3, .7)
        self.loss_components = {}

    def forward(self, outputs, targets):
        assert 'stage_P' in outputs and 'stage_Q' not in outputs
        seg = self.seg_loss(outputs['logits'], targets)
        r_total = seg.new_zeros(())
        smooth_total = seg.new_zeros(())
        comps = {}
        for i, (p, weight) in enumerate(zip(outputs['stage_P'], [.4, .3, .2, .1])):
            r = amplitude(p)
            mask = F.interpolate(targets.float().unsqueeze(1), size=p.shape[-2:], mode='nearest')
            fg = (mask > .5).float()
            bg = 1 - fg
            rf = ((1 - r).square() * fg).sum() / fg.sum().clamp_min(1)
            rb = (r.square() * bg).sum() / bg.sum().clamp_min(1)
            smooth = polar_smoothness(p, mask)
            r_total = r_total + weight * (rf + rb)
            smooth_total = smooth_total + weight * smooth
            comps.update({
                f'P/stage{i}/fg_loss': rf.item(),
                f'P/stage{i}/bg_loss': rb.item(),
                f'P/stage{i}/fg_mean': ((r * fg).sum() / fg.sum().clamp_min(1)).item(),
                f'P/stage{i}/bg_mean': ((r * bg).sum() / bg.sum().clamp_min(1)).item(),
                f'P_smooth/stage{i}': (weight * smooth).item(),
            })
        nl = outputs.get('cr_losses', {})
        nem = seg.new_zeros(())
        bottleneck = seg.new_zeros(())
        for stage in [2, 3]:
            if torch.is_tensor(nl.get(f'loss_nem{stage}')):
                nem = nem + nl[f'loss_nem{stage}']
            if torch.is_tensor(nl.get(f'loss_bottleneck{stage}')):
                bottleneck = bottleneck + nl[f'loss_bottleneck{stage}']
        field = r_total + smooth_total
        total = seg + .3 * field + .03 * nem + .05 * bottleneck
        self.loss_components = {
            'Total': total.item(),
            'Seg': seg.item(),
            'P_w': (.3 * field).item(),
            'P_amplitude': r_total.item(),
            'P_smooth': smooth_total.item(),
            'NIBS_nem': (.03 * nem).item(),
            'NIBS_b': (.05 * bottleneck).item(),
            **comps,
        }
        return total


def get_loss_function(*args, **kwargs):
    return NativePolarLoss()
