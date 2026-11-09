"""Data / checkpoint loading with the README-11 interface (--beta --zdim --pca).

Truth (theta) is returned in a separate object so that callers can keep the
unsupervised search code physically separated from post-hoc evaluation.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Use THIS repo's modlearn_rot (an older checkout may be pip-installed).
sys.path.insert(0, str(ROOT / "np" / "src"))

D_T = {"C1": 4, "C2": 4, "C3": 6, "CR3": 4}


def number_tag(v: float) -> str:
    from modlearn_rot.training.continuous import number_tag as nt
    return nt(v)


def condition_tag(beta, zdim, lambda_sparse=0.0, cmin=0.25, cmax=0.80):
    from modlearn_rot.training.continuous import training_condition_tag
    return training_condition_tag(beta, lambda_sparse, cmin, cmax, zdim)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def add_common_args(ap):
    ap.add_argument("--model", required=True, choices=["C1", "C2", "C3", "CR3"])
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--zdim", type=int, required=True)
    ap.add_argument("--pca", type=int, choices=[0, 1], required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=20261108)
    return ap


def paths(model, beta, zdim, repeat=1):
    cond = condition_tag(beta, zdim)
    base = ROOT / "outputs" / "choice" / model
    return dict(
        condition=cond,
        latent=base / "s4_evaluate" / cond / f"repeat{repeat:02d}" / "s4_evaluate__latent.npz",
        checkpoint=base / "s3_train" / cond / f"repeat{repeat:02d}" / "s3_train__checkpoint.pt",
        dataset=base / "s1_generate" / "s1_generate__dataset.npz",
    )


class Truth:
    """Container for ground truth; only post-hoc / oracle code may touch it."""

    def __init__(self, theta, train_ids):
        self.theta = np.asarray(theta, float)
        self.mu = self.theta[train_ids].mean(0)
        self.sd = self.theta[train_ids].std(0)

    def std(self, ids=None):
        t = (self.theta - self.mu) / self.sd
        return t if ids is None else t[ids]


def load_all(model, beta, zdim, pca, repeat=1, need_checkpoint=True):
    p = paths(model, beta, zdim, repeat)
    for k in ("latent", "dataset") + (("checkpoint",) if need_checkpoint else ()):
        if not p[k].exists():
            raise FileNotFoundError(f"missing {k}: {p[k]}")
    lat = np.load(p["latent"], allow_pickle=False)
    d_t = D_T[model]
    z = np.asarray(lat["z"], np.float64)
    if z.shape[1] != zdim:
        raise RuntimeError("zdim mismatch with latent file")
    if pca == 0 and zdim != d_t:
        raise RuntimeError("--pca 0 requires zdim == d_T")
    if pca == 1 and zdim < d_t:
        raise RuntimeError("--pca 1 requires zdim >= d_T")
    if pca == 1 and zdim > d_t:
        raise RuntimeError("zdim > d_T with pca=1 needs a residual protocol; not implemented")
    if need_checkpoint and str(lat["checkpoint_sha256"]) != sha256(p["checkpoint"]):
        raise RuntimeError("checkpoint sha mismatch with latent file")
    if str(lat["context_mode"]) != "all_observed_trials":
        raise RuntimeError("latent was not extracted from all observed trials")
    tr, va, te = (np.asarray(lat[k]) for k in ("train_ids", "validation_ids", "test_ids"))
    mu_z = z[tr].mean(0)
    zc = z - mu_z
    if pca == 1:
        cov = np.cov(zc[tr].T)
        ev, evec = np.linalg.eigh(cov)
        order = np.argsort(ev)[::-1]
        C = evec[:, order[:d_t]].T  # k x zdim
        C *= np.sign(C[np.arange(d_t), np.abs(C).argmax(1)])[:, None]
    else:
        C = np.eye(zdim)
    V = zc @ C.T  # coordinates that are rotated: U = R V
    data = np.load(p["dataset"], allow_pickle=False)
    out = dict(model=model, condition=p["condition"], paths={k: str(v) for k, v in p.items() if k != "condition"},
               z=z, mu_z=mu_z, C=C, V=V, train_ids=tr, val_ids=va, test_ids=te,
               x=np.asarray(data["x"], np.float64), y=np.asarray(data["y"]).reshape(-1),
               subject_index=np.asarray(data["subject_index"]),
               predictive_true=np.asarray(data["predictive_true"]).reshape(-1))
    truth = Truth(np.asarray(data["parameters"], np.float64), tr)
    if not np.allclose(truth.theta, np.asarray(lat["theta_true"]), atol=1e-8):
        raise RuntimeError("theta in dataset and latent disagree")
    return out, truth


def load_decoder(checkpoint_path, device):
    import torch
    from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig, DiscreteNeuralProcess
    try:
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        ck = torch.load(checkpoint_path, map_location="cpu")
    cfg = DiscreteNPConfig(**ck["model_config"])
    net = DiscreteNeuralProcess(cfg)
    net.load_state_dict(ck["model_state_dict"], strict=True)
    net = net.to(device).eval()
    net.requires_grad_(False)
    return net, cfg


def pick_device(name):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)
