"""Exact empirical whole-U/joint-X grid ALS for separated functions.

Sufficient statistics factor across the two whole blocks, never across
dependent scalar coordinates within a block. Each flattened model is
independent. A free global intercept is profiled, not a free U/X function.
"""
import numpy as np
import torch


class GridALS:
    def __init__(self, u_phi, x_phi, active_u, active_x, targets, groups, channels,
                 starts, rank, seed=408, previous=None):
        self.u_phi, self.x_phi = u_phi, x_phi
        self.active_u, self.active_x = active_u, active_x
        self.targets = targets  # channels, N, M; each has mean zero
        self.groups, self.channels, self.starts, self.rank = groups, channels, starts, rank
        device = u_phi.device
        self.group_index = torch.arange(groups, device=device).repeat(starts*channels)
        self.channel_index = torch.arange(channels, device=device).repeat_interleave(groups).repeat(starts)
        count, p = len(self.group_index), u_phi.shape[-1]
        generator = torch.Generator().manual_seed(seed+rank*1000)
        def factors(d):
            template = torch.randn(starts, channels, d, p, rank, generator=generator, dtype=torch.float64)*(.6/np.sqrt(p-1))
            template[..., 0, :] = 1.
            return template[:, :, None].expand(starts, channels, groups, d, p, rank).reshape(count, d, p, rank).clone().to(device)
        self.u, self.x = factors(u_phi.shape[2]), factors(x_phi.shape[1])
        self.amplitude = torch.ones(count, rank, dtype=torch.float64, device=device)*.01
        if previous is not None:
            old = previous['amplitude'].shape[-1]
            self.u[..., :old] = previous['u']; self.x[..., :old] = previous['x']
            self.amplitude[:, :old] = previous['amplitude']; self.amplitude[:, old:] = 0.
        self.normalize()

    def normalize(self):
        for name, active in (('u', self.active_u[self.group_index]), ('x', self.active_x.expand(len(self.x), -1, -1))):
            factor = getattr(self, name)
            factor *= active[..., None]
            norm = torch.sqrt(torch.clamp((factor.square()*active[..., None]).sum(2), min=1e-30))
            factor /= norm[:, :, None]
            if name == 'u':
                # Initial factors are normalized without changing initial alpha;
                # the warm-start factors already have unit marginal norm.
                pass

    def snapshot(self):
        return dict(u=self.u.clone(), x=self.x.clone(), amplitude=self.amplitude.clone())

    def products(self, ids):
        u = torch.einsum('lndp,ldpr->lndr', self.u_phi[self.group_index[ids]], self.u[ids])
        x = torch.einsum('mdp,ldpr->lmdr', self.x_phi, self.x[ids])
        return u, x

    def target_product(self, factors, ids, transpose=False):
        # Avoid replicating the complete N*M target for every independent model.
        result = torch.empty(len(ids), self.targets.shape[2 if transpose else 1], self.rank,
                             dtype=torch.float64, device=self.u.device)
        channels = self.channel_index[ids]
        for c in range(self.channels):
            selected = torch.nonzero(channels == c, as_tuple=True)[0]
            if len(selected):
                packed = factors[selected].permute(1, 0, 2).reshape(factors.shape[1], -1)
                target = self.targets[c].T if transpose else self.targets[c]
                product = target @ packed
                result[selected] = product.reshape(len(product), len(selected), self.rank).permute(1, 0, 2)
        return result

    def update(self, block, axis, ridge=1e-8, model_chunk=8):
        count = len(self.u); n, m = self.targets.shape[1:]
        for first in range(0, count, model_chunk):
            ids = torch.arange(first, min(first+model_chunk, count), device=self.u.device)
            fu, fx = self.products(ids)
            if block == 'u':
                other = torch.cat((fu[:, :, :axis], fu[:, :, axis+1:]), 2).prod(2)
                phi = self.u_phi[self.group_index[ids], :, axis]
                opposite = fx.prod(2)
                target_opposite = self.target_product(opposite, ids)
                active = self.active_u[self.group_index[ids], axis]
            else:
                other = torch.cat((fx[:, :, :axis], fx[:, :, axis+1:]), 2).prod(2)
                phi = self.x_phi[None, :, axis].expand(len(ids), -1, -1)
                opposite = fu.prod(2)
                target_opposite = self.target_product(opposite, ids, transpose=True)
                active = self.active_x[axis].expand(len(ids), -1)
            design = torch.einsum('lnr,lnp->lnrp', other, phi)
            p = phi.shape[-1]; length = design.shape[1]
            flat = design.reshape(len(ids), length, -1)
            cross = opposite.transpose(1, 2) @ opposite/opposite.shape[1]
            gram = (flat.transpose(1, 2) @ flat/length).reshape(len(ids), self.rank, p, self.rank, p)
            gram = (gram*cross[:, :, None, :, None]).reshape(len(ids), self.rank*p, self.rank*p)
            mean = (design.mean(1)*opposite.mean(1)[:, :, None]).reshape(len(ids), -1)
            gram -= mean[:, :, None]*mean[:, None, :]
            rhs = torch.einsum('lnrp,lnr->lrp', design, target_opposite).reshape(len(ids), -1)/(n*m)
            scale = torch.clamp(torch.diagonal(gram, dim1=1, dim2=2).mean(1), min=1e-15)
            system = (gram+gram.transpose(1, 2))/2+ridge*scale[:, None, None]*torch.eye(self.rank*p, device=gram.device)
            coefficient = torch.linalg.solve(system, rhs[..., None])[..., 0].reshape(len(ids), self.rank, p).transpose(1, 2)
            coefficient *= active[:, :, None]
            norm = torch.sqrt(torch.clamp(coefficient.square().sum(1), min=1e-30))
            getattr(self, block)[ids, axis] = coefficient/norm[:, None]
            self.amplitude[ids] = norm

    def loss(self, model_chunk=8):
        result = torch.empty(len(self.u), dtype=torch.float64, device=self.u.device)
        n, m = self.targets.shape[1:]
        target_energy = self.targets.square().mean((1, 2))
        for first in range(0, len(self.u), model_chunk):
            ids = torch.arange(first, min(first+model_chunk, len(self.u)), device=self.u.device)
            fu, fx = self.products(ids); u, x = fu.prod(2), fx.prod(2)
            gu = u.transpose(1, 2) @ u/n; gx = x.transpose(1, 2) @ x/m
            amplitude = self.amplitude[ids]
            predicted_energy = torch.einsum('lr,ls,lrs,lrs->l', amplitude, amplitude, gu, gx)
            predicted_mean = (amplitude*u.mean(1)*x.mean(1)).sum(1)
            rhs = (u*self.target_product(x, ids)).sum(1)/(n*m)
            error = target_energy[self.channel_index[ids]]-2*(amplitude*rhs).sum(1)+predicted_energy-predicted_mean.square()
            if not torch.isfinite(error).all() or error.min() < -1e-7:
                raise ArithmeticError('Grid contraction is nonfinite or has negative squared error')
            result[ids] = torch.clamp(error, min=0.)
        return result

    def predict_grid(self, model_id):
        ids = torch.tensor([model_id], device=self.u.device)
        fu, fx = self.products(ids); u, x = fu[0].prod(1), fx[0].prod(1)
        prediction = (u*self.amplitude[model_id]) @ x.T
        return prediction-prediction.mean()
