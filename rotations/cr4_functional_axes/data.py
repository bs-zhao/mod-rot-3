"""Suite-local S1/S3/S5 loader; no simulator theta is accessed until evaluation."""
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_source(args, config):
    root = ROOT / "outputs" / "choice" / args.model
    ck = root / "s3_train" / args.condition / f"repeat{args.repeat:02d}" / "s3_train__checkpoint.pt"
    latent = root / "s4_evaluate" / args.condition / f"repeat{args.repeat:02d}" / "s4_evaluate__latent.npz"
    dataset = root / "s1_generate" / "s1_generate__dataset.npz"
    for f in (ck, latent, dataset):
        if not f.is_file():
            raise FileNotFoundError(f"Missing NP source: {f}")
    ds_sha, ck_sha, latent_sha = sha256(dataset), sha256(ck), sha256(latent)
    with np.load(latent, allow_pickle=False) as a:
        if str(a["dataset_sha256"]) != ds_sha or str(a["checkpoint_sha256"]) != ck_sha:
            raise RuntimeError("S5 Z does not match S1 dataset and S3 checkpoint")
        if str(a["context_mode"]) != "all_observed_trials":
            raise RuntimeError("Rotation requires full-observed-trial S5 posterior mean")
        z = np.asarray(a["z"], dtype=np.float64)
        tr = np.asarray(a["train_ids"], dtype=np.int64)
        va = np.asarray(a["validation_ids"], dtype=np.int64)
        te = np.asarray(a["test_ids"], dtype=np.int64)
    if not np.array_equal(np.sort(np.concatenate((tr, va, te))), np.arange(len(z))):
        raise ValueError("S5 subject split is not disjoint and exhaustive")
    k = int(args.rotation_dim)
    if k not in ((3,4) if args.model=="CR3" else (3,)) or z.shape[1]<k:
        raise ValueError(f"{args.model}: CR3 permits rotation_dim 3 or 4; "
                         f"CR4/CR5 require 3. Native-Z dim={z.shape[1]}")
    pca = PCA(n_components=k, svd_solver="full").fit(z[tr])
    w = pca.components_ / np.sqrt(np.maximum(pca.explained_variance_, config["whiten_floor"]))[:, None]
    b = np.linalg.pinv(w)
    q = (z - pca.mean_) @ w.T
    residual = z - pca.mean_ - q @ b.T
    if not np.allclose(z, pca.mean_ + q @ b.T + residual, atol=1e-9):
        raise RuntimeError("PCA residual roundtrip failed")
    with np.load(dataset, allow_pickle=False) as a:
        counts = np.asarray(a["trial_counts"], dtype=np.int64)
        if len(counts) != len(z) or not np.all(counts == 500):
            raise ValueError("CR3/CR4/CR5 require 500 trials per S5 subject")
        # A fixed X bank is shared by all people. Read-only, no Y or theta.
        bank = np.asarray(a["x"][:500], dtype=np.float64)
        ids = np.asarray(a["subject_index"], dtype=np.int64)
        if not np.array_equal(ids, np.repeat(np.arange(len(z)), 500)):
            raise ValueError("S1 subject order differs from S5")
        if not np.all(np.isfinite(bank)):
            raise ValueError("Invalid bank X")
        if not np.array_equal(np.asarray(a["x"][-500:]), bank):
            raise ValueError("Shared bank does not match last participant")
    rng = np.random.default_rng(config["seed"])
    fs = np.sort(rng.choice(tr, size=min(len(tr), config["fit_subjects"]), replace=False))
    vs = np.sort(rng.choice(va, size=min(len(va), config["valid_subjects"]), replace=False))
    ix = rng.permutation(len(bank))
    nx = min(len(bank), int(config["fit_x"]))
    nv = min(len(bank) - nx, int(config["valid_x"]))
    if len(fs) < 8 or len(vs) < 8 or nx < 8 or nv < 8:
        raise ValueError("Need >=8 FIT/VALID persons and X anchors")
    fit_x = np.sort(ix[:nx])
    valid_x = np.sort(ix[nx:nx+nv])
    return dict(root=root, model=args.model, checkpoint=ck, dataset=dataset,
                latent=latent, hashes=dict(checkpoint=ck_sha,dataset=ds_sha,latent=latent_sha),
                z=z, q=q, residual=residual, W=w, B=b, pca_mean=pca.mean_.copy(),
                tr=tr, va=va, te=te, fit_subject_ids=fs, valid_subject_ids=vs,
                bank=bank, fit_x_ids=fit_x, valid_x_ids=valid_x)
