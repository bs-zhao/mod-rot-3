"""Support-screened finite effects and conditional direction consistency."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def clusters(z, count, seed=383):
    """Euclidean clustering; signed coordinate permutations preserve labels."""
    rng = np.random.default_rng(seed)
    centers = z[rng.choice(len(z), min(count, len(z)), replace=False)].copy()
    labels = np.zeros(len(z), dtype=int)
    for _ in range(25):
        ds = np.sum((z[:, None]-centers[None])**2, axis=2)
        updated = ds.argmin(1)
        if np.array_equal(updated, labels):
            break
        labels = updated
        for k in range(len(centers)):
            if np.any(labels == k):
                centers[k] = z[labels == k].mean(0)
    return labels


def direction_loss(effect, labels=None):
    """Cancellation in effect[n, stimulus, axis], allowing a sign per context.

    Zero effects contribute zero weight. This is an empirical local sign
    diagnostic, not a proof of monotonicity over an unobserved domain.
    """
    if labels is None:
        signed = np.abs(effect.sum(0)).sum()
    else:
        signed = 0.
        for k in np.unique(labels):
            signed += np.abs(effect[labels == k].sum(0)).sum()
    absolute = np.abs(effect).sum()
    return float(max(0., absolute-signed)/max(absolute, 1e-24))


def effect_scores(effect, contexts):
    out = {"fixed_X_direction_inconsistency": direction_loss(effect)}
    for count, labels in contexts.items():
        numerator = denominator = 0.
        for i, label in enumerate(labels):
            e = effect[:, :, i:i+1]
            weight = float(np.abs(e).sum())
            numerator += direction_loss(e, label)*weight
            denominator += weight
        out[f"conditional_direction_k{count}"] = numerator/max(denominator, 1e-24)
    energy = effect**2
    ordered = np.sort(energy, axis=-1)[..., ::-1]
    total = energy.sum(-1)
    # Mean pointwise normalized tails: distinct from pooled-energy tails.
    for k in (1, 2):
        tails = np.sqrt(ordered[..., k:].sum(-1)/np.maximum(total, 1e-24))
        out[f"pointwise_axis_tail{k}_l2"] = float(np.mean(tails[total > 1e-20])) if np.any(total > 1e-20) else 0.
    return out


def support_mask(u, maps, steps=(.2, .5), factor=1.5):
    """One common observed-anchor subset for ALL directions, cases and scales.

    A shifted point's 16-neighbor radius must not exceed factor times its
    anchor's radius. This is a geometric diagnostic, not a density
    theorem or proof of interpolation adequacy.
    """
    tree = cKDTree(u)
    neighbors = min(16, len(u))
    radius = tree.query(u, k=neighbors)[0][:, -1]
    keep = np.ones(len(u), dtype=bool)
    for q in maps:
        for i in range(u.shape[1]):
            for step in steps:
                for sign in (-1, 1):
                    shifted = u+sign*step/2*q[:, i]
                    distances = tree.query(shifted, k=neighbors)[0][:, -1]
                    keep &= distances <= factor*np.maximum(radius, 1e-12)
    return keep


def contexts(u, counts=(8, 32)):
    return {count: [clusters(np.delete(u, i, axis=1), count)
                    for i in range(u.shape[1])] for count in counts}


def ux_contrast_scores(gradient, x):
    """Effects of observed stimulus contrasts, not partial X derivatives."""
    scale = np.std(x, axis=0); scale[scale < 1e-10] = 1.
    z = x/scale
    _, neighbors = cKDTree(z).query(z, k=min(4, len(z)))
    edges = sorted({tuple(sorted((i, int(j)))) for i in range(len(z))
                    for j in neighbors[i, 1:] if i != j})
    if not edges:
        return {"UX_contrast_tail1_l2": 0., "UX_contrast_tail2_l2": 0.}
    a, b = np.asarray(edges).T
    delta = gradient[:, a]-gradient[:, b]
    energy = np.mean(delta**2, axis=0)
    ordered = np.sort(energy, axis=-1)[:, ::-1]
    total = energy.sum(-1)
    return {f"UX_contrast_tail{k}_l2": float(np.mean(np.sqrt(
        ordered[:, k:].sum(-1)/np.maximum(total, 1e-24)))) for k in (1, 2)}
