# Frozen decoder read utilities copied from mlp_teacher_rotation/teacher.py at 2d89e4e for suite independence.
"""Frozen NP decoder as teacher: soft targets and exact Z-derivative fingerprints.

This suite does not train, fine-tune or update the NP. True simulator theta is
not read here. Decoder X is RAW, as in DiscreteNPTrainer; only the encoder
receives normalized observations.
"""
from __future__ import annotations
import numpy as np
import torch
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig, DiscreteNeuralProcess
from modlearn_rot.training.continuous import load_checkpoint


def _device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _restore(source, config):
    ck = load_checkpoint(source["checkpoint_path"])
    model_config = DiscreteNPConfig(**ck["model_config"])
    if model_config.latent_dim != source["z"].shape[1]:
        raise RuntimeError("Decoder latent dimension differs from S5 Z")
    if model_config.x_dim != source["x_fit"].shape[-1] or model_config.num_choices != 2:
        raise RuntimeError("Decoder X dimension or binary response family mismatch")
    device = _device(str(config["teacher_device"]))
    model = DiscreteNeuralProcess(model_config)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model = model.to(device).eval()
    model.requires_grad_(False)
    return model, device


def direct_fingerprints(model, device, source, config):
    """Autodiff d(logit difference)/dq and d(probability)/dq at common FIT X anchors.

    q is whitened TRAIN PCA; dz/dq_i = sqrt(eigenvalue_i) * PCA_component_i.
    Each subject retains its original Z residual outside the chosen subspace.
    """
    z = np.asarray(source["z_fit"], np.float32)
    x = np.asarray(source["x_fit"], np.float32)
    mask = np.asarray(source["mask_fit"], bool)
    n, k = len(z), int(source["rotation_dim"])
    ns = min(n, int(config["direct_anchor_subjects"]))
    # For CR3's shared fixed bank, sample unique trial rows from ONE subject.
    # For independently generated C1/C2/C3 X, pool FIT observations instead.
    fixed = bool(n > 1 and np.array_equal(mask[0],mask[1]) and
                 np.array_equal(x[0,mask[0]],x[1,mask[1]]))
    pool = x[0,mask[0]] if fixed else x[mask]
    nx = min(len(pool), int(config["direct_anchor_x"]))
    if ns < 8 or nx < 4:
        raise ValueError("Direct fingerprint anchors require >=8 subjects and >=4 X")
    rng = np.random.default_rng(int(config["seed"]) + 773)
    subjects = np.sort(rng.choice(n, ns, replace=False))
    trial_ids = np.sort(rng.choice(len(pool), nx, replace=False))
    common_x = np.asarray(pool[trial_ids], np.float32)
    del pool
    pca = source["pca"]
    # The whitening floor must match the original rotation data loader.
    scales = np.sqrt(np.maximum(np.asarray(pca.explained_variance_[:k]),
                                float(source["whiten_floor"])))
    directions = np.asarray(pca.components_[:k] * scales[:, None], np.float32)
    gp = np.zeros((ns, k, nx), np.float64)
    gl = np.zeros_like(gp)
    bs = int(config["direct_batch_subjects"])
    if bs < 1:
        raise ValueError("direct_batch_subjects must be positive")
    common = torch.tensor(common_x, dtype=torch.float32, device=device)
    for begin in range(0, ns, bs):
        end = min(ns, begin + bs)
        zb = torch.tensor(z[subjects[begin:end]], dtype=torch.float32, device=device)
        xb = common.unsqueeze(0).expand(len(zb), -1, -1)

        def logodds(latent):
            logits = model.decode(xb, latent)
            return logits[..., 1] - logits[..., 0]

        # Forward-mode AD is exact for the frozen GELU MLP (up to float precision).
        for axis in range(k):
            tangent = torch.tensor(directions[axis], device=device).expand_as(zb)
            logit, dlogit = torch.func.jvp(logodds, (zb,), (tangent,))
            p = torch.sigmoid(logit)
            gl[begin:end, axis] = dlogit.detach().cpu().numpy()
            gp[begin:end, axis] = (dlogit*p*(1-p)).detach().cpu().numpy()
    if not np.isfinite(gp).all() or not np.isfinite(gl).all():
        raise RuntimeError("Non-finite frozen decoder derivatives")
    return dict(probability=gp, logit=gl, common_x=common_x,
                anchor_fit_subject_ids=subjects, dz_dq=directions,
                differentiation="torch.func.jvp; original Z with intact PCA residual",
                anchor_pool_mode=("shared_fixed_bank" if fixed else "pooled_fit_x"))



