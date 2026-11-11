"""Build subject-by-response profiles exclusively from Step-2 W and common Y.

No formula, theta, best R or decoder weights are read. For a repeated,
identical stimulus bank use one output value per stimulus and subject.
Otherwise estimate each person's response coefficients against a generic
constant + linear + quadratic X basis. The same profiles are used for ALL
candidate rotated W matrices.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CHANNELS = ("logit_nn", "p_nn", "choice_nn", "choice_observed")


def _read_row_mapping(stage2):
    folder = Path(stage2)
    sid = np.load(folder / "subject_index.npy", allow_pickle=False)
    tid = np.load(folder / "trial_index.npy", allow_pickle=False)
    counts = np.load(folder / "trial_counts.npy", allow_pickle=False)
    n = len(counts)
    m = int(counts.sum())
    if (n < 8 or m < n or sid.shape != (m,) or tid.shape != (m,)
            or np.any(counts <= 0)):
        raise ValueError("Subject/trial metadata have inconsistent shapes")
    if not np.array_equal(sid, np.repeat(np.arange(n), counts)):
        raise ValueError("subject_index must list contiguous, consecutive subject blocks")
    offsets = np.r_[0, np.cumsum(counts)]
    for i in range(n):
        a, b = int(offsets[i]), int(offsets[i+1])
        if not np.array_equal(tid[a:b], np.arange(b-a)):
            raise ValueError(f"Nonconsecutive trial IDs within subject {i}")
    return counts, sid, tid, offsets


def _shared_bank(x, counts, offsets):
    """Infer repeated X design from raw numeric X, without using model names."""
    if not np.all(counts == counts[0]):
        return False
    bank = np.asarray(x[:counts[0]])
    for s in range(1, len(counts)):
        a, b = int(offsets[s]), int(offsets[s+1])
        if not np.array_equal(x[a:b], bank):
            return False
    return True


def _features(x):
    """Design-independent polynomial basis, degree <= 2, in original X."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    cols = [np.ones(n)]
    names = ["intercept"]
    for i in range(d):
        cols.append(x[:,i]);names.append(f"x{i+1}")
    for i in range(d):
        cols.append(x[:,i]**2);names.append(f"x{i+1}^2")
    for i in range(d):
        for j in range(i+1,d):
            cols.append(x[:,i]*x[:,j]);names.append(f"x{i+1}*x{j+1}")
    return np.column_stack(cols), names


def load_profiles(step2, first_blind_case, channels=CHANNELS,
                  ridge=1e-3):
    """Return identical F[channel][N, p] for each W case, with metadata."""
    step2 = Path(step2)
    counts, sid, tid, offsets = _read_row_mapping(step2)
    n, m = len(counts), len(sid)
    w = np.load(first_blind_case / "W.npy", mmap_mode="r", allow_pickle=False)
    if w.ndim != 2 or w.shape[0] != m:
        raise ValueError("One Step-2 W must have one row per original trial")
    index = json.loads((step2/"index.json").read_text(encoding="utf-8"))
    dz = int(index["Z_dim"])
    dx = int(index["X_dim"])
    if w.shape[1] != dz+dx:
        raise ValueError("W width differs from Step-2 index")
    x = w[:, dz:]
    with np.load(step2/"Y.npz", allow_pickle=False) as y:
        response = {}
        for ch in channels:
            if ch not in CHANNELS:
                raise ValueError(f"Unsupported channel: {ch}")
            if ch not in y:
                raise ValueError(f"Shared Y is missing {ch}")
            value = np.asarray(y[ch], dtype=np.float64)
            if value.shape != (m,) or not np.isfinite(value).all():
                raise ValueError(f"Y[{ch}] is not finite M-vector")
            response[ch] = value
    shared = _shared_bank(x, counts, offsets)
    result = {}
    if shared:
        for ch in channels:
            result[ch] = response[ch].reshape(n, int(counts[0]))
        names = [f"stimulus_{i+1}" for i in range(int(counts[0]))]
        kind = "shared_stimulus_bank"
    else:
        # A fixed polynomial X basis is a *measurement convention*, not
        # an assumption about the hidden behavioral generating equation.
        # Standardization is learned from all raw X rows, independent of W.
        mean = np.asarray(np.mean(x, axis=0, dtype=np.float64))
        std = np.asarray(np.std(x, axis=0, dtype=np.float64))
        if (std < 1e-12).any():
            raise ValueError("Non-repeated X design contains constant X dimensions")
        normed = False
        for s in range(n):
            a,b = int(offsets[s]),int(offsets[s+1])
            phi, names = _features((np.asarray(x[a:b],dtype=np.float64)-mean)/std)
            yy = np.column_stack([response[ch][a:b] for ch in channels])
            gram = phi.T @ phi
            reg = np.eye(len(gram)) * float(ridge)
            reg[0,0] = 0.
            coef = np.linalg.solve(gram+reg,phi.T@yy)
            if not normed:
                for ch in channels:
                    result[ch] = np.empty((n, len(names)), dtype=np.float64)
                normed = True
            for j,ch in enumerate(channels):
                result[ch][s] = coef[:,j]
        kind = "per_subject_degree2_X_response_coefficients"
    for ch, f in result.items():
        if f.ndim != 2 or f.shape[0] != n or not np.isfinite(f).all():
            raise ValueError(f"Invalid response profile for {ch}")
    meta = {
        "profile_kind": kind,
        "n_subjects": n,
        "n_trials": m,
        "n_profile_features": len(names),
        "profile_feature_names": names,
        "n_latent_axes": dz,
        "n_x_features": dx,
        "channels": list(channels),
        "subject_profile_ridge": float(ridge) if not shared else None,
        "input_data_allowed": "Step2 trial W (X only); shared Y; subject/trial IDs",
        "true_theta_used": False,
        "formula_used": False,
        "decoder_weights_used": False,
        "profile_X_design_inferred_from_matrix": True,
        "note": (
            "In repeated-bank mode, response profile has one column per actual "
            "stimulus. In non-repeated mode, coefficients summarize each "
            "individual's dependence on a fixed low-degree X basis. "
            "Profiles can be imperfect summaries; examine sensitivity."
        ),
    }
    return result, meta, (counts, sid, tid, offsets)


def profile_views(profile, kind):
    """Two response structures: full output and X-dependent contrasts."""
    profile = np.asarray(profile, dtype=np.float64)
    if kind == "shared_stimulus_bank":
        return {"full": profile, "contrast": profile - profile.mean(axis=1,keepdims=True)}
    return {"full": profile, "contrast": profile[:,1:]}
