"""One radial GP fit; all subsequent coordinate changes are exact rotations."""
import numpy as np
import torch

import gp_ard_complexity as gp
import gp_posterior_effects as posterior


def model(u, targets, center_ids, starts=3):
    result = gp.Model(u, targets, center_ids, starts)
    # Equal initial eta entries plus identical projected gradients keep each
    # model's eta exactly radial throughout the existing optimizer.
    result.eta.register_hook(lambda gradient: gradient.mean(1, keepdim=True).expand_as(gradient))
    return result


@torch.no_grad()
def gradient_profiles(fitted, ids, rows):
    phi, target, _, ratio, _, _, solved = fitted.system(ids, rows)
    beta = ratio[:, None, None]*solved
    groups = fitted.group_index[ids]; u = fitted.u[groups][:, rows]
    centers = fitted.u[groups][:, fitted.center_ids]; eta = fitted.eta[ids]
    if float((eta-eta[:, :1]).abs().max()) > 1e-12:
        raise ValueError('This rotation-covariant source must remain isotropic')
    scaled = centers*eta[:, None]
    cc_distance = torch.clamp(scaled.square().sum(-1)[:, :, None]+scaled.square().sum(-1)[:, None]-2*scaled @ scaled.transpose(1, 2), min=0.)
    kcc = torch.exp(-.5*cc_distance)
    eye = torch.eye(len(fitted.center_ids), dtype=torch.float64, device=u.device)
    chol = torch.linalg.cholesky((kcc+kcc.transpose(1, 2))/2+1e-6*eye)
    alpha = torch.linalg.solve_triangular(chol.transpose(1, 2), beta, upper=True)
    difference = u[:, :, None]-centers[:, None]
    kernel = torch.exp(-.5*(difference*eta[:, None, None]).square().sum(-1))
    gradients = torch.stack([(-eta[:, None, i:i+1].square()*difference[..., i]*kernel) @ alpha
                             for i in range(u.shape[-1])], dim=2)
    error = (phi @ beta-target).square().sum((1, 2))/torch.clamp(target.square().sum((1, 2)), min=1e-30)
    return gradients, error


@torch.no_grad()
def covariances(fitted, ids, rows):
    gradients, error = gradient_profiles(fitted, ids, rows)
    covariance = torch.einsum('bnir,bnjr->bnij', gradients, gradients)
    return covariance.cpu().numpy(), error.cpu().numpy()


def energies(covariance, q):
    """covariance[subject,X,U,U] -> energy[subject,rotated-U,X]."""
    return np.maximum(np.einsum('ai,nxab,bi->nix', q, covariance, q, optimize=True), 0.)


def scores(covariance, q):
    return posterior.scores(energies(covariance, q))


def primary_value(covariance, q):
    e = energies(covariance, q)
    peak = e.max(1, keepdims=True)
    e = np.divide(e, peak, out=np.zeros_like(e), where=peak > 0.)
    total = e.sum(1, keepdims=True)
    p = np.divide(e, total, out=np.zeros_like(e), where=total > 0.)
    return float(np.mean(np.sum(p/(p+.1**2), axis=1)))
