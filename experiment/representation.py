"""Native signed-vector field throughout the model. No P-to-Q decoding."""

import hashlib
import math
import types

import numpy as np
import torch
import torch.nn.functional as F


def state_hash(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode())
        result.update(str(tuple(tensor.shape)).encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def amplitude(p):
    return torch.sqrt(p[:, 0:1].square() + p[:, 1:2].square() + 1e-8)


def bounded(p):
    return p * torch.clamp(1.0 / (amplitude(p) + 1e-8), max=1.0)


def resample(p, size):
    return bounded(F.interpolate(p, size=size, mode='bilinear', align_corners=False))


@torch.no_grad()
def image_prior(self, image):
    """Direct energy-weighted first angular moment of signed image tangents.

    P = local_mean(|grad I| * (-Iy, Ix)) / local_mean(|grad I|^2).
    This averages signed unit tangents with gradient-energy weights. Its native
    norm is polar order; no second-angle field or axis-lifting step is used.
    """
    z = image.float()
    lo = z.amin(dim=(-2, -1), keepdim=True)
    hi = z.amax(dim=(-2, -1), keepdim=True)
    z = (z - lo) / (hi - lo).clamp_min(1e-6)
    if z.shape[1] > 1:
        z = z.mean(dim=1, keepdim=True)
    tool = self.p_image_prior
    z = F.conv2d(z, tool.kernel_g, padding=tool.kernel_g.shape[-1] // 2)
    gx, gy = tool._sobel(z)
    weight = torch.sqrt(gx.square() + gy.square())
    stats = torch.cat([-weight * gy, weight * gx, weight.square()], dim=1)
    stats = F.conv2d(stats, tool.kernel_t.expand(3, -1, -1, -1),
                     padding=tool.kernel_t.shape[-1] // 2, groups=3)
    p = bounded(stats[:, :2] / (stats[:, 2:3] + 1e-8))
    if self._axis_group == 'P_AUG' and self.training:
        signs = np.where(
            self._axis_rng.random((p.shape[0], 1, *self._axis_settings['sign_grid']))
            < self._axis_settings['sign_probability'], -1., 1.)
        p = p * F.interpolate(torch.as_tensor(signs, device=p.device, dtype=p.dtype),
                              size=p.shape[-2:], mode='nearest')
    elif self._axis_eval_mode == 'global_flip':
        p = -p
    elif self._axis_eval_mode == 'alternate_halfplane':
        p = p * torch.where(p[:, 1:2] < 0,
                            -torch.ones_like(p[:, 1:2]), torch.ones_like(p[:, 1:2]))
    elif self._axis_eval_mode == 'quadrants':
        p = p * F.interpolate(p.new_tensor([[-1., 1.], [1., -1.]])[None, None],
                              size=p.shape[-2:], mode='nearest')
    return torch.cat([p, amplitude(p)], dim=1)


def polar_responses(gradient, p):
    """Signed contributions to parallel and perpendicular directional derivatives.

    Their pairwise sums equal n_x*d_x+n_y*d_y and -n_y*d_x+n_x*d_y.
    Two channels per feature retain the original branch widths and parameters.
    Below the original reliability threshold, use the original isotropic fallback.
    """
    r = amplitude(p)
    reliable = (r > .05).to(p.dtype)
    nx = p[:, 0:1] / r.clamp_min(1e-6)
    ny = p[:, 1:2] / r.clamp_min(1e-6)
    rx, ry = gradient[:, :, 0], gradient[:, :, 1]
    par = torch.cat([reliable * nx * rx + (1 - reliable) * .5 * rx,
                     reliable * ny * ry + (1 - reliable) * .5 * ry], dim=1)
    perp = torch.cat([-reliable * ny * rx + (1 - reliable) * .5 * rx,
                      reliable * nx * ry + (1 - reliable) * .5 * ry], dim=1)
    return par, perp


def polar_conv(self, x, p):
    b, c, h, w = x.shape
    gradient = self.grad(x).view(b, c, 2, h, w)
    par, perp = polar_responses(gradient, p)
    return x + self.fusion(torch.cat([self.branch_par(par), self.branch_perp(perp)], dim=1))


def refine(self, x, p):
    pred = self.p_pred(torch.cat([x, p], dim=1))
    pred = bounded(pred)
    gate = torch.sigmoid(self.scale) * self.conv(x)
    out = p + gate * (pred - p)
    if self.stats_logger is not None:
        self.stats_logger.log_q_update(self.name, gate.mean().item(),
                                       torch.sigmoid(self.scale).item(),
                                       amplitude(out).mean().item())
    return out, {}


def transition(self, x, p):
    out = self.feat_down(x)
    return out, resample(p, out.shape[-2:])


def encoder(self, x, prior):
    h = self.stem(torch.cat([x, prior], dim=1))
    p = resample(prior[:, :2], h.shape[-2:])
    stem = p
    h, p, features, fields, nibs = self.backbone(h, p, s_raw=prior[:, 2:3])
    return features, fields, stem, nibs


def nofm(self, x, p=None, *args, **kwargs):
    x = x + self.drop_path(self.ssm(self.norm_ssm(x)))
    r = amplitude(p) if p is not None else None
    return x + self.drop_path(self.nglu(self.norm_glu(x), r))


def forward(self, x):
    normalized = self.input_norm(x)
    prior = self._stage_prior(x)
    features, fields, stem, nibs = self.encoder(normalized, prior)
    return {'logits': self.decoder(features, normalized),
            'stage_P': fields, 'cr_losses': nibs}


def install(model, group, settings, seed):
    assert group in ['P', 'P_AUG']
    model._axis_group = group
    model._axis_eval_mode = 'normal'
    model._axis_settings = settings
    model._axis_rng = np.random.default_rng(int(seed) + settings['sign_seed_offset'])
    model.p_image_prior = model.q_image_prior
    del model.q_image_prior
    model._stage_prior = types.MethodType(image_prior, model)
    model.forward = types.MethodType(forward, model)
    model.encoder.forward = types.MethodType(encoder, model.encoder)
    for updater in model.encoder.q_updaters:
        updater.p_pred = updater.q_pred
        del updater.q_pred
        updater.forward = types.MethodType(refine, updater)
    for layer in model.encoder.backbone.transitions:
        layer.forward = types.MethodType(transition, layer)
    for stage in model.encoder.backbone.stages:
        for block in stage:
            if hasattr(block, 'nematic'):
                block.nematic.forward = types.MethodType(polar_conv, block.nematic)
            if hasattr(block, 'nglu'):
                block.forward = types.MethodType(nofm, block)
    return model


def forbid_q_runtime():
    """Fail immediately if any legacy Q consumer remains on the native P path."""
    from networks.qtensor import QTensorTools, QImagePrior

    def forbidden(*args, **kwargs):
        raise RuntimeError('Native P path called a Q operation')

    for name in ['get_s', 'get_qtensor', 'get_nematic',
                 'downsample_q', 'get_unit_q', 'get_director_numpy']:
        setattr(QTensorTools, name, staticmethod(forbidden))
    QImagePrior.forward = forbidden
