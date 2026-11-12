"""Actual posterior-mean function changes, separate from GP ARD priors."""
import numpy as np
import torch

import conditional_ux
import local_support


@torch.no_grad()
def effects(model, ids, rows, steps=(.05, .2)):
    phi, target, _, ratio, _, _, solved = model.system(ids, rows)
    beta = ratio[:, None, None]*solved
    groups = model.group_index[ids]
    u = model.u[groups][:, rows]
    centers = model.u[groups][:, model.center_ids]
    eta2 = model.eta[ids].square()
    distance_cc = torch.clamp(((centers[:, :, None]-centers[:, None])*model.eta[ids, None, None]).square().sum(-1), min=0.)
    kcc = torch.exp(-.5*distance_cc)
    eye = torch.eye(len(model.center_ids), dtype=torch.float64, device=u.device)
    chol = torch.linalg.cholesky((kcc+kcc.transpose(1, 2))/2+1e-6*eye)
    alpha = torch.linalg.solve_triangular(chol.transpose(1, 2), beta, upper=True)
    difference = u[:, :, None]-centers[:, None]
    distance = (difference.square()*eta2[:, None, None]).sum(-1)
    kernel = torch.exp(-.5*distance)
    fields = {'derivative': []}
    fields.update({f'finite_{step:g}': [] for step in steps})
    for i in range(u.shape[-1]):
        derivative = -eta2[:, None, i:i+1]*difference[..., i]*kernel
        fields['derivative'].append((derivative @ alpha).square().sum(-1))
        for step in steps:
            h = step/2
            increment = 2*h*eta2[:, None, i:i+1]*difference[..., i]
            shift = h*h*eta2[:, None, i:i+1]
            plus = torch.exp(-.5*torch.clamp(distance+increment+shift, min=0.))
            minus = torch.exp(-.5*torch.clamp(distance-increment+shift, min=0.))
            fields[f'finite_{step:g}'].append((((plus-minus)/step) @ alpha).square().sum(-1))
    energies = {name: torch.stack(values, dim=-1).cpu().numpy() for name, values in fields.items()}
    fidelity = (phi @ beta-target).square().sum((1, 2))/torch.clamp(target.square().sum((1, 2)), min=1e-30)
    return energies, fidelity.cpu().numpy()


def scores(energy):
    """energy[subject,Uaxis,Xcolumn], normalizing before or after pooling."""
    result = {'pooled_'+name: value for name, value in conditional_ux.sparsity(energy.mean(0)).items()}
    for metric, values in local_support.row_scores(energy.transpose(0, 2, 1)).items():
        for x_label, by_subject in (('Xmean', values.mean(1)), ('Xworst', values.max(1))):
            result[f'point_UX_columns_{x_label}_subject_mean_{metric}'] = float(by_subject.mean())
            result[f'point_UX_columns_{x_label}_subject_median_{metric}'] = float(np.median(by_subject))
    return result
