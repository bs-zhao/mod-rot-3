"""Low separated-rank function regression, with independent batched models."""
import numpy as np
import torch


def features(x):
    x = np.asarray(x, dtype=float)
    knots = np.arange(-2., 3.)
    return np.concatenate((np.ones_like(x)[..., None], x[..., None], x[..., None]**2,
        x[..., None]**3, x[..., None]**4, np.exp(-.5*(x[..., None]-knots)**2)), axis=-1)


def whiten(phi):
    """Preserve constant 1, center and whiten remaining univariate features."""
    single = phi.ndim == 3
    if single:
        phi = phi[None]
    groups, n, d, p = phi.shape
    mean = phi[..., 1:].mean(axis=1)
    centered = phi[..., 1:]-mean[:, None]
    covariance = np.einsum('gndp,gndq->gdpq', centered, centered, optimize=True)/n
    values, vectors = np.linalg.eigh(covariance)
    active = values > 1e-10*np.maximum(values.max(axis=-1, keepdims=True), np.finfo(float).tiny)
    inverse = np.zeros_like(values); inverse[active] = 1/np.sqrt(values[active])
    transform = vectors*inverse[..., None, :]
    result = np.concatenate((np.ones((groups, n, d, 1)),
        np.einsum('gndp,gdpq->gndq', centered, transform, optimize=True)), axis=-1)
    active = np.concatenate((np.ones((groups, d, 1), dtype=bool), active), axis=-1)
    if single:
        return result[0], mean[0], transform[0], active[0]
    return result, mean, transform, active


def canonical_classes(u, maps, tolerance=1e-5):
    """Remove U signs/permutations by W-only canonicalization, no case labels."""
    canonical, membership = [], []
    for q in maps:
        z = u @ q
        signs = []
        for j in range(z.shape[1]):
            nonzero = np.flatnonzero(np.abs(z[:, j]) > 1e-10)
            signs.append(float(np.sign(z[nonzero[0], j])) if len(nonzero) else 1.)
        order = sorted(range(z.shape[1]), key=lambda j: tuple(np.abs(z[:8, j])))
        qc = q[:, order]*np.asarray(signs)[order]
        found = next((k for k, previous in enumerate(canonical) if np.linalg.norm(previous-qc) < tolerance), None)
        if found is None:
            found = len(canonical); canonical.append(qc)
        membership.append(found)
    return np.asarray(canonical), np.asarray(membership)


class Model(torch.nn.Module):
    """Every leading-axis model has separate parameters and gradients."""
    def __init__(self, groups, channels, starts, d, dx, p, rank, device='cpu', seed=404, previous=None):
        super().__init__()
        self.groups, self.channels, self.starts, self.rank = groups, channels, starts, rank
        # Flatten start, channel, coordinate class. Initial templates repeat
        # across classes, so a case's fit does not depend on its list position.
        self.group_index = torch.arange(groups, device=device).repeat(starts*channels)
        self.channel_index = torch.arange(channels, device=device).repeat_interleave(groups).repeat(starts)
        count = starts*channels*groups
        generator = torch.Generator(device='cpu').manual_seed(seed+rank*1000)
        def factors(dimension):
            template = torch.randn((starts, channels, dimension, p, rank), generator=generator)*(.3/np.sqrt(p-1))
            template[:, :, :, 0, :] = 1.
            return template[:, :, None].expand(starts, channels, groups, dimension, p, rank).reshape(count, dimension, p, rank).clone().to(device)
        self.u = torch.nn.Parameter(factors(d)); self.x = torch.nn.Parameter(factors(dx))
        template = torch.randn((starts, channels, rank), generator=generator)*.01
        self.amplitude = torch.nn.Parameter(template[:, :, None].expand(starts, channels, groups, rank).reshape(count, rank).clone().to(device))
        self.bias = torch.nn.Parameter(torch.zeros(count, device=device))
        if previous is not None:
            r = previous['amplitude'].shape[-1]
            with torch.no_grad():
                self.u[..., :r].copy_(previous['u']); self.x[..., :r].copy_(previous['x'])
                self.amplitude[:, :r].copy_(previous['amplitude']); self.amplitude[:, r:] = 0.
                self.bias.copy_(previous['bias'])

    def effective(self, model_ids, active_u, active_x, dtype=torch.float32):
        group = self.group_index[model_ids]
        wu, wx = self.u[model_ids].to(dtype), self.x[model_ids].to(dtype)
        nu = torch.sqrt(torch.clamp((wu.square()*active_u[group, :, :, None]).sum(2), min=1e-8))
        nx = torch.sqrt(torch.clamp((wx.square()*active_x[None, :, :, None]).sum(2), min=1e-8))
        return wu/nu[:, :, None], wx/nx[:, :, None], self.amplitude[model_ids].to(dtype), self.bias[model_ids].to(dtype)

    def predict(self, model_ids, subjects, x_phi, u_phi, active_u, active_x, dtype=torch.float32):
        group = self.group_index[model_ids]
        up = u_phi[group[:, None], subjects[None, :]].to(dtype)
        wu, wx, amplitude, bias = self.effective(model_ids, active_u, active_x, dtype)
        fu = torch.einsum('lbdp,ldpr->lbdr', up, wu).prod(2)
        fx = torch.einsum('bdp,ldpr->lbdr', x_phi.to(dtype), wx).prod(2)
        return torch.sum(fu*fx*amplitude[:, None], dim=-1)+bias[:, None]

    def snapshot(self):
        return {key: value.detach().clone() for key, value in self.named_parameters()}


def clip_independent_gradients(model, maximum=5.):
    squared = sum(p.grad.square().reshape(len(p), -1).sum(1) for p in model.parameters())
    multiplier = torch.clamp(maximum/torch.sqrt(squared+1e-24), max=1.)
    for p in model.parameters():
        p.grad.mul_(multiplier.reshape(len(p), *([1]*(p.ndim-1))))
