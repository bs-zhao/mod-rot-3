"""Active monotone-piece complexity of conditional Ui/Xj response slices."""
import numpy as np
import torch
from scipy.spatial import cKDTree


class Evaluator:
    def __init__(self, meta, coefficients, device='cuda'):
        self.meta = meta; self.device = device
        self.coef = torch.as_tensor(coefficients, dtype=torch.float64, device=device)
        self.blocks = []
        if meta['kind'] == 'poly':
            powers = meta['powers']; lookup = {tuple(p): i for i, p in enumerate(powers)}
            for order in range(1, int(powers.sum(1).max())+1):
                ids = np.flatnonzero(powers.sum(1) == order); parents, variables = [], []
                for row in ids:
                    p = powers[row].copy(); j = int(np.flatnonzero(p)[0]); p[j] -= 1
                    parents.append(lookup[tuple(p)]); variables.append(j)
                self.blocks.append((ids, torch.tensor(parents, device=device), torch.tensor(variables, device=device)))
        else:
            self.centers = torch.as_tensor(meta['centers'], dtype=torch.float64, device=device)

    @torch.no_grad()
    def __call__(self, probes):
        z = torch.as_tensor(probes, dtype=torch.float64, device=self.device)
        if self.meta['kind'] == 'poly':
            phi = torch.empty(len(z), len(self.coef), dtype=torch.float64, device=self.device); phi[:, 0] = 1.
            for ids, parents, variables in self.blocks:
                phi[:, int(ids[0]):int(ids[-1])+1] = phi.index_select(1, parents)*z.index_select(1, variables)
        else:
            distance = torch.clamp(z.square().sum(1)[:, None]+self.centers.square().sum(1)[None]-2*z @ self.centers.T, min=0.)
            kernel = torch.exp(-distance/(2*self.meta['lengthscale']**2))
            phi = torch.cat((torch.ones((len(z), 1), dtype=torch.float64, device=z.device), kernel), dim=1)
        return phi @ self.coef


def conditional_intervals(z, q, neighbors=64, quantiles=(.1, .9)):
    u = z @ q; bounds = np.empty((len(z), u.shape[1], 2))
    for i in range(u.shape[1]):
        rest = np.delete(u, i, axis=1)
        ids = cKDTree(rest).query(rest, k=min(neighbors, len(z)))[1]
        bounds[:, i] = np.quantile(u[ids, i], quantiles, axis=1).T
    return u, bounds


def probes_for(u, bounds, q, nodes):
    n, d = u.shape
    values = bounds[..., :1]+np.linspace(0., 1., nodes)[None, None]*(bounds[..., 1:]-bounds[..., :1])
    probes = np.broadcast_to(u[:, None, None, :], (n, d, nodes, d)).copy()
    for i in range(d):
        probes[:, i, :, i] = values[:, i]
    return probes.reshape(-1, d) @ q.T


@torch.no_grad()
def piece_costs(values, widths, epsilons=(.01, .03, .1)):
    """values[subject,channel,Xcolumn,Uaxis,node,recorded-X-context].

    One monotone active direction costs one piece. Reversals add fractional
    pieces by total variation / range. Soft activation uses effect energy
    relative to the mean context energy of this subject/X-column field.
    This preserves arbitrary positive scaling of the whole X-column effect
    while giving nearly flat contexts little activation.
    """
    centered = values-values[:, :, :, :, values.shape[4]//2:values.shape[4]//2+1]
    peak = centered.abs().amax(dim=(3, 4, 5), keepdim=True)
    unit = torch.where(peak > 0, centered/torch.clamp(peak, min=torch.finfo(values.dtype).tiny), 0.)
    amplitude = unit.amax(4)-unit.amin(4)
    variation = torch.diff(unit, dim=4).abs().sum(4)
    width = torch.as_tensor(widths, dtype=values.dtype, device=values.device)[:, None, None, :, None]
    slopes = torch.where(width > 0, amplitude/torch.clamp(width, min=torch.finfo(values.dtype).tiny), 0.)
    maximum = slopes.amax(dim=(3, 4), keepdim=True)
    energy = torch.where(maximum > 0, slopes/torch.clamp(maximum, min=torch.finfo(values.dtype).tiny), 0.).square()
    reference = energy.sum(3, keepdim=True).mean(4, keepdim=True)
    pieces = torch.where(amplitude > 0, variation/torch.clamp(amplitude, min=torch.finfo(values.dtype).tiny), 0.)
    result = {}
    for eps in epsilons:
        denominator = energy+eps**2*reference
        active = torch.where(denominator > 0, energy/torch.clamp(denominator, min=torch.finfo(values.dtype).tiny), 0.)
        label = f'eps{int(eps*1000):03d}'
        result[label] = (active*pieces).sum(3).mean(-1).cpu().numpy()
        result['active_'+label] = active.sum(3).mean(-1).cpu().numpy()
    return result


def evaluate_case(z, q, evaluator, channels, dx, contexts, nodes=17, neighbors=64,
                  subjects_per_chunk=32, support_k=16, support_factor=1.5):
    u, bounds = conditional_intervals(z, q, neighbors)
    tree = cKDTree(z); radii = tree.query(z, k=min(support_k, len(z)))[0][:, -1]
    tables, covered, total = {}, 0, 0
    for start in range(0, len(z), subjects_per_chunk):
        stop = min(start+subjects_per_chunk, len(z)); n, d = stop-start, z.shape[1]
        probes = probes_for(u[start:stop], bounds[start:stop], q, nodes)
        distances = tree.query(probes, k=min(support_k, len(z)))[0][:, -1].reshape(n, d, nodes)
        covered += int(np.sum(distances <= support_factor*radii[start:stop, None, None])); total += distances.size
        values = evaluator(probes).reshape(n, d, nodes, channels, dx, contexts).permute(0, 3, 4, 1, 2, 5)
        partial = piece_costs(values, bounds[start:stop, :, 1]-bounds[start:stop, :, 0])
        if not tables:
            tables = {key: np.empty((len(z), channels, dx)) for key in partial}
        for key, value in partial.items():
            tables[key][start:stop] = value
    return tables, dict(probe_coverage=covered/total, probed_U_locations=total,
        all_subjects=len(z), nodes=nodes, conditional_neighbors=neighbors,
        interval_width_minimum=float(np.min(bounds[..., 1]-bounds[..., 0])),
        interval_width_median=float(np.median(bounds[..., 1]-bounds[..., 0])))


def scores(tables, channel):
    result = {}
    for key, table in tables.items():
        active = key.startswith('active_')
        family = 'UX_conditional_active_count' if active else 'UX_conditional_monotone_pieces'
        label = key[7:] if active else key
        for x_label, values in (('mean', table[:, channel].mean(1)), ('worst', table[:, channel].max(1))):
            result[f'{family}_X{x_label}_subject_mean_{label}'] = float(values.mean())
            result[f'{family}_X{x_label}_subject_median_{label}'] = float(np.median(values))
    return result
