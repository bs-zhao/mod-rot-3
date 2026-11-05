"""Subject alignment and held-out auditing, separate from theta-free plugin fitting."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment


def aligned_observations(data):
    """Construct padded subject-aligned X, Y, mask; mask excludes padding."""
    counts = np.asarray(data.trial_counts, dtype=int)
    if counts.ndim != 1 or (counts < 1).any():
        raise ValueError('Each subject must have at least one trial')
    n = len(counts)
    t = int(counts.max())
    d = int(data.x.shape[1])
    x = np.zeros((n, t, d), dtype=np.float32)
    y = np.zeros((n, t), dtype=np.int64)
    valid = np.zeros((n, t), dtype=bool)
    offsets = np.r_[0, np.cumsum(counts)]
    if int(offsets[-1]) != len(data.x) or not np.array_equal(
        np.asarray(data.subject_index), np.repeat(np.arange(n), counts)
    ):
        raise ValueError('Raw X/Y are not in contiguous, aligned subject order')
    for i in range(n):
        begin, end = int(offsets[i]), int(offsets[i+1])
        x[i, :counts[i]] = data.x[begin:end]
        y[i, :counts[i]] = np.asarray(data.y[begin:end]).reshape(-1)
        valid[i, :counts[i]] = True
    return x, y, valid


def corr(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    denominator = np.sqrt(np.sum(a*a, axis=0)[:, None] * np.sum(b*b, axis=0)[None, :])
    return np.divide(a.T @ b, denominator, out=np.zeros((a.shape[1], b.shape[1])),
                     where=denominator > 1e-12)


def evaluate_after_freeze(theta, u, test_ids, folder: Path, method: str, parameter_names):
    """Oracle/theta exclusively after method result and fit metadata are saved."""
    c = corr(np.asarray(theta)[test_ids], u[test_ids])
    p, k = c.shape
    rows, cols = linear_sum_assignment(-c*c)
    matched = {int(r): int(col) for r, col in zip(rows, cols)}
    chosen = np.asarray([c[r, j]**2 for r, j in matched.items()])
    off = np.ones_like(c, dtype=bool)
    for r, j in matched.items():
        off[r, j] = False
    report = {
        'method': method,
        'matched_mean_r2': float(chosen.mean()) if len(chosen) else None,
        'matched_min_r2': float(chosen.min()) if len(chosen) else None,
        'n_matched': len(matched),
        'parameter_r2': {str(parameter_names[i]): (float(c[i, matched[i]]**2) if i in matched else None)
                         for i in range(p)},
        'max_offtarget_abs_r': float(np.abs(c[off]).max()) if off.any() else 0.0,
        'mean_offtarget_r2': float((c[off]**2).mean()) if off.any() else 0.0,
        'matching': matched,
        'theta_axis_corr': c.tolist(),
        'rows': list(parameter_names),
        'columns': [f'u{j+1}' for j in range(k)],
    }
    (folder / 'rotation__test_evaluation.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(max(6, k * .78), max(4.0, p * .68)))
    im = ax.imshow(c**2, vmin=0, vmax=1, cmap='Blues')
    ax.set_yticks(range(p), list(parameter_names))
    ax.set_xticks(range(k), [f'u{j+1}' for j in range(k)])
    for i in range(p):
        for j in range(k):
            ax.text(j, i, f'{c[i, j]**2:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if c[i, j]**2 > .55 else 'black')
    from matplotlib.patches import Rectangle
    for i, j in matched.items():
        ax.add_patch(Rectangle((j-.5, i-.5), 1, 1, fill=False, edgecolor='red', lw=2))
    fig.colorbar(im, ax=ax, label='TEST Pearson r²')
    fig.tight_layout()
    fig.savefig(folder/'rotation__test_corr2.png', dpi=170)
    plt.close(fig)
    return report