"""Frozen four-model benchmark data and coordinate-matched analytic reference.

No training, no changes to existing experiments. The map from V to standardized
true parameters is learned on TRAIN only; it is a supervised *diagnostic*, not
a ground-truth identity and not an unsupervised rotation.
"""
from __future__ import annotations
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from three_gaps.common import io, analytic


DEFAULTS = {
    "C1": (0.0005, 4), "C2": (0.0005, 4),
    "C3": (0.0005, 6), "CR3": (0.00005, 4),
}


@dataclass
class Case:
    model: str
    repeat: int
    data: dict
    truth: object
    net: torch.nn.Module
    A: np.ndarray
    b: np.ndarray
    v_mu: np.ndarray
    v_sd: np.ndarray
    x_mu: np.ndarray
    x_sd: np.ndarray
    ids: np.ndarray
    V: np.ndarray
    Tstd: np.ndarray
    X: np.ndarray
    checkpoint_sha: str
    grid: str

    @property
    def d(self):
        return self.V.shape[1]

    @property
    def p(self):
        return self.X.shape[-1]


def load_case(model, repeat=1, n_subjects=32, n_stimuli=64,
              seed=20261116, grid="shared", device="cpu",
              beta=None, zdim=None, pca=0):
    if model not in DEFAULTS:
        raise ValueError(model)
    default_beta, default_dim = DEFAULTS[model]
    beta = default_beta if beta is None else beta
    zdim = default_dim if zdim is None else zdim
    data, truth = io.load_all(model, beta, zdim, pca, repeat)
    net, _ = io.load_decoder(data["paths"]["checkpoint"], torch.device(device))
    tr, te = data["train_ids"], data["test_ids"]
    # Matched coordinates: Tstd ≈ A V + b. Never fit using evaluation IDs.
    design = np.column_stack((data["V"][tr], np.ones(len(tr))))
    coefficients = np.linalg.lstsq(design, truth.std(tr), rcond=None)[0]
    A, b = coefficients[:-1].T, coefficients[-1]
    v_mu = np.mean(data["V"][tr], axis=0)
    v_sd = np.std(data["V"][tr], axis=0).clip(1e-10)
    x_train = data["x"][np.isin(data["subject_index"], tr)]
    x_mu = np.mean(x_train, axis=0)
    x_sd = np.std(x_train, axis=0).clip(1e-10)
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(te, min(n_subjects, len(te)), replace=False))
    V, Tstd = data["V"][ids].copy(), truth.std(ids)
    if grid == "shared":
        if model == "CR3":
            bank = data["x"][data["subject_index"] == 0]
            if n_stimuli <= 0 or n_stimuli >= len(bank):
                X = bank.copy()
            else:
                indices = np.sort(rng.choice(len(bank), n_stimuli, replace=False))
                X = bank[indices]
        else:
            k = n_stimuli if n_stimuli > 0 else 400
            k0 = (k + 1) // 2
            values = rng.uniform(-1, 1, (k0, analytic.X_DIM[model]))
            X = np.concatenate((values, -values), axis=0)[:k]
    elif grid == "paired":
        idx = data["subject_index"]
        rows = [data["x"][idx == s][:n_stimuli] for s in ids]
        if not rows or min(map(len, rows)) < n_stimuli:
            raise ValueError("not enough paired trials per subject")
        X = np.stack(rows)
    else:
        raise ValueError("grid must be shared or paired")
    return Case(model, repeat, data, truth, net, A, b, v_mu, v_sd,
                x_mu, x_sd, ids, V, Tstd, X,
                io.sha256(data["paths"]["checkpoint"]), grid)


def input_pairs(case, standardized=False, reference="mapped"):
    """Return coordinates of shape (S,d) and (K,p) or (S,K,p)."""
    if reference == "truth":
        v = case.Tstd
        u_mu = np.zeros(case.d)
        u_sd = np.ones(case.d)
    else:
        v = case.V
        u_mu, u_sd = case.v_mu, case.v_sd
    x = case.X
    if standardized:
        v = (v-u_mu)/u_sd
        x = (x-case.x_mu)/case.x_sd
    return np.asarray(v, dtype=np.float64), np.asarray(x, dtype=np.float64)


def make_function(case, source, channel="logit", precision="float64",
                  standardized=False):
    """Return differentiable h(u,x) in the selected *common* input coordinates.

    source=analytic_mapped uses TRAIN-fit linear Tstd(V), with no subject lookup.
    source=analytic_truth takes U=Tstd and is *not* comparable pointwise to NN.
    source=nn uses the frozen full decoder and exact inverse V->Z mapping.
    """
    if channel not in ("logit", "prob"):
        raise ValueError(channel)
    if source not in ("analytic_mapped", "analytic_truth", "nn"):
        raise ValueError(source)
    dtype = torch.float64 if precision == "float64" else torch.float32
    net = copy.deepcopy(case.net).to(dtype=dtype).eval() if source == "nn" else None
    if net is not None:
        net.requires_grad_(False)
    if source == "analytic_truth":
        uv_mu = np.zeros(case.d)
        uv_sd = np.ones(case.d)
    else:
        uv_mu, uv_sd = case.v_mu, case.v_sd
    def h(u, x):
        if standardized:
            u = u*torch.as_tensor(uv_sd, device=u.device, dtype=u.dtype) + torch.as_tensor(uv_mu, device=u.device, dtype=u.dtype)
            x = x*torch.as_tensor(case.x_sd, device=x.device, dtype=x.dtype) + torch.as_tensor(case.x_mu, device=x.device, dtype=x.dtype)
        if source == "analytic_truth":
            tstd = u
        elif source == "analytic_mapped":
            A = torch.as_tensor(case.A, device=u.device, dtype=u.dtype)
            b = torch.as_tensor(case.b, device=u.device, dtype=u.dtype)
            tstd = A @ u + b
        else:
            z = (torch.as_tensor(case.data["mu_z"], device=u.device, dtype=u.dtype) +
                 torch.as_tensor(case.data["Minv"], device=u.device, dtype=u.dtype) @ u)
            logits = net.decoder(torch.cat((x, z)).to(dtype))
            eta = (logits[1]-logits[0]).to(u.dtype)
            return eta if channel == "logit" else torch.sigmoid(eta)
        T = (torch.as_tensor(case.truth.mu, device=u.device, dtype=u.dtype) +
             torch.as_tensor(case.truth.sd, device=u.device, dtype=u.dtype)*tstd)
        eta = analytic.ETA[case.model](T, x)
        return eta if channel == "logit" else torch.sigmoid(eta)
    return h


def strict_x_interior(case, X):
    """Conservative flag for differentiation in all X axes, not a validity test."""
    if case.model != "CR3":
        return np.ones(X.shape[:-1], dtype=bool)
    a = np.asarray(X)
    amounts = a[..., [0, 1, 3, 4]]
    probs = a[..., [2, 5]]
    return np.all(amounts > 1e-8, axis=-1) & np.all(
        (probs > 1e-8) & (probs < 1-1e-8), axis=-1)


def metadata(case):
    return {"model": case.model, "repeat": case.repeat,
            "subjects": len(case.ids), "stimuli": len(case.X[0] if case.X.ndim==3 else case.X),
            "grid": case.grid, "subject_ids": case.ids.tolist(),
            "beta_zdim": DEFAULTS[case.model], "checkpoint_sha256": case.checkpoint_sha,
            "A_train": case.A.tolist(), "b_train": case.b.tolist(),
            "heldout_linear_map_r2": (1-np.var(case.Tstd-(case.V@case.A.T+case.b), axis=0)/
                                    np.var(case.Tstd, axis=0).clip(1e-12)).tolist()}


def save_metadata(path, case):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path/"case.json").write_text(json.dumps(metadata(case), indent=2), encoding="utf-8")
