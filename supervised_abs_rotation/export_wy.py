"""Export Step-1 subject/trial arrays using the *frozen* original NP decoder.

W=[Zstd_subject_repeated, X_original], whereas decoder inputs are
[X_original, Zraw_subject_repeated] -- exactly the NP training convention.
Never pass z-scored Z into a decoder trained on raw latent posterior means.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NP_SRC = ROOT / "np" / "src"


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for piece in iter(lambda: f.read(1024 * 1024), b""):
            h.update(piece)
    return h.hexdigest()


def load_original_trials(dataset_path, n_subjects):
    """Read S1's exact trial order (including variable counts in C1/C2/C3)."""
    with np.load(dataset_path, allow_pickle=False) as f:
        keys = {"trial_counts", "subject_index", "trial_index", "x", "y", "parameters"}
        if not keys.issubset(f.files):
            raise ValueError(f"S1 dataset is missing keys: {sorted(keys - set(f.files))}")
        counts = np.asarray(f["trial_counts"], dtype=np.int64)
        sid = np.asarray(f["subject_index"], dtype=np.int64)
        tid = np.asarray(f["trial_index"], dtype=np.int64)
        x = np.asarray(f["x"], dtype=np.float32)
        y_raw = np.asarray(f["y"]).reshape(-1)
        theta_n = int(f["parameters"].shape[0])
    if counts.shape != (n_subjects,) or theta_n != n_subjects:
        raise ValueError("S1 subject count does not match frozen S5 Z")
    if counts.min() <= 0 or x.ndim != 2 or x.shape[1] <= 0:
        raise ValueError("Expected a positive trial count and a trialwise X matrix")
    m = int(counts.sum())
    if sid.shape != (m,) or tid.shape != (m,) or x.shape[0] != m or len(y_raw) != m:
        raise ValueError("S1 trial arrays do not have a shared total-trials dimension")
    if not np.array_equal(sid, np.repeat(np.arange(n_subjects), counts)):
        raise ValueError("S1 trial ordering must consist of contiguous subject blocks")
    expected_tid = np.arange(m) - np.repeat(
        np.r_[0, np.cumsum(counts)[:-1]], counts)
    if not np.array_equal(tid, expected_tid):
        raise ValueError("Trial indices do not match original S1 within-subject ordering")
    if not np.isin(y_raw, [0, 1]).all():
        raise ValueError("Expected original observed binary 0/1 choices")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite S1 X data")
    return counts, sid, tid, x, y_raw.astype(np.uint8)


def load_frozen_decoder(checkpoint_path, dataset_path, z_dim, x_dim, device):
    """Restore exactly the S3 decoder, with no retraining or re-encoding."""
    if str(NP_SRC) not in sys.path:
        sys.path.insert(0, str(NP_SRC))
    import torch
    from modlearn_rot.nn.np.point.discrete import (
        DiscreteNPConfig, DiscreteNeuralProcess)
    from modlearn_rot.training.continuous import load_checkpoint

    ckpt = load_checkpoint(checkpoint_path)
    config = DiscreteNPConfig(**ckpt["model_config"])
    if config.x_dim != x_dim or config.latent_dim != z_dim or config.num_choices != 2:
        raise ValueError(
            f"Frozen decoder geometry differs from S1/S5: "
            f"checkpoint X={config.x_dim}, Z={config.latent_dim}, C={config.num_choices}; "
            f"observed X={x_dim}, Z={z_dim}")
    expected_sha = ckpt.get("extra_state", {}).get("dataset_sha256")
    if expected_sha != file_sha256(dataset_path):
        raise ValueError("Frozen S3 checkpoint belongs to another S1 dataset")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    model = DiscreteNeuralProcess(config).to(device_obj)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, device_obj


def export_stage1_data(*, output, dataset_path, checkpoint_path,
                       z_raw, z_std, r, model_name,
                       batch_size=8192, device="auto",
                       overwrite=False, source_provenance=None):
    """Write a standalone subfolder with Zstd, R, W, binary NN Y and indices.

    Output folder: output/stage1_data/
       Zstd.npy: (N,dz) posterior mean standardized across ALL subjects
       R.npy: (dz,dz) supervised orthogonal transform; U=Zstd @ R.T
       W.npy: (M,dz+dx) [Zstd[sid], X_original]
       Y.npz: (M,) original observed 0/1, binary logit, p1, decoder argmax,
               plus two raw class-logit columns (M,2)
       subject_index.npy, trial_index.npy, trial_counts.npy, Zraw.npy,
       manifest.json: layout, provenance, leakage warning, decoder convention.

    The decoder receives original S5 Z and original X. It must NOT use R or
    Zstd because the frozen NN was trained on [X, original Z].
    """
    if batch_size < 1:
        raise ValueError("decoder batch size must be positive")
    z_raw = np.asarray(z_raw, dtype=np.float64)
    z_std = np.asarray(z_std, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if z_raw.ndim != 2 or z_std.shape != z_raw.shape:
        raise ValueError("Expected matching (N,d) original and standardized Z")
    n, dz = z_raw.shape
    if r.shape != (dz, dz) or not np.allclose(r @ r.T, np.eye(dz), atol=1e-7):
        raise ValueError("R must be orthogonal and match Z dimensions")
    if not np.isfinite(z_raw).all() or not np.isfinite(z_std).all():
        raise ValueError("Non-finite latent values")
    expected_zstd = (z_raw - z_raw.mean(0)) / z_raw.std(0, ddof=0)
    if not np.allclose(z_std, expected_zstd, rtol=1e-7, atol=1e-7):
        raise ValueError("Zstd must be computed from all subjects (ddof=0)")

    dataset_path = Path(dataset_path)
    checkpoint_path = Path(checkpoint_path)
    counts, sid, tid, x, choice_observed = load_original_trials(dataset_path, n)
    m, dx = x.shape
    model, device_obj = load_frozen_decoder(
        checkpoint_path, dataset_path, dz, dx, device)

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "stage1_data"
    if target.exists() and not overwrite:
        raise FileExistsError(f"Step-1 trial export exists; pass --overwrite: {target}")

    import torch
    with tempfile.TemporaryDirectory(prefix=".stage1_data_build_", dir=out) as temporary:
        stage = Path(temporary) / "stage1_data"
        stage.mkdir()
        np.save(stage / "Zstd.npy", z_std)
        np.save(stage / "Zraw.npy", z_raw.astype(np.float32))
        np.save(stage / "R.npy", r)
        np.save(stage / "subject_index.npy", sid)
        np.save(stage / "trial_index.npy", tid)
        np.save(stage / "trial_counts.npy", counts)

        w = np.lib.format.open_memmap(
            stage / "W.npy", mode="w+", dtype=np.float32, shape=(m, dz + dx))
        logit = np.empty(m, dtype=np.float32)
        p1 = np.empty(m, dtype=np.float32)
        choice_decoder = np.empty(m, dtype=np.uint8)
        two_logits = np.empty((m, 2), dtype=np.float32)

        with torch.inference_mode():
            for start in range(0, m, batch_size):
                end = min(start + batch_size, m)
                batch_sid = sid[start:end]
                xb = x[start:end]
                zb = z_raw[batch_sid].astype(np.float32)
                w[start:end, :dz] = z_std[batch_sid].astype(np.float32)
                w[start:end, dz:] = xb
                decoder_input = np.concatenate([xb, zb], axis=1)
                logits = model.decoder(
                    torch.from_numpy(decoder_input).to(device_obj))
                if logits.shape != (end-start, 2):
                    raise RuntimeError(f"Unexpected binary decoder output: {logits.shape}")
                if not bool(torch.isfinite(logits).all()):
                    raise ValueError("Frozen decoder produced NaN/Inf logits")
                scores = logits.detach().cpu().numpy()
                two_logits[start:end] = scores
                logit[start:end] = scores[:, 1] - scores[:, 0]
                p1[start:end] = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                choice_decoder[start:end] = np.argmax(scores, axis=1).astype(np.uint8)
        w.flush()
        del w

        if not np.isfinite(logit).all() or not np.isfinite(p1).all():
            raise ValueError("Non-finite decoder logit/probability")
        if not np.all((p1 >= 0) & (p1 <= 1)):
            raise ValueError("Decoder probability is outside [0,1]")
        # Preserve original choice distinct from decoder argmax; no sampling.
        np.savez_compressed(
            stage / "Y.npz",
            choice_observed=choice_observed,
            logit_nn=logit,
            p_nn=p1,
            choice_nn=choice_decoder,
            logits_nn_2class=two_logits,
        )
        manifest = {
            "model": model_name,
            "n_subjects": int(n),
            "n_trials_total": m,
            "n_trials_min": int(counts.min()),
            "n_trials_max": int(counts.max()),
            "z_dim": dz,
            "x_dim": dx,
            "w_shape": [m, dz+dx],
            "W_columns": [f"zstd{i+1}" for i in range(dz)]
                         + [f"x{i+1}" for i in range(dx)],
            "W_dtype": "float32",
            "Zstd_dtype": "float64",
            "R_dtype": "float64",
            "row_order": "EXACT original S1 trial order, contiguous subject blocks",
            "subject_index": "0-based; matches original S1 subject_index",
            "trial_index": "0-based; matches original S1 trial_index",
            "Zstd": "full-subject S5 posterior means standardized per latent axis with ddof=0",
            "R_application": "U_subject = Zstd @ R.T; R is exported separately and is NOT applied to W",
            "W": "hstack([Zstd[subject_index], X_original])",
            "decoder_input": "hstack([X_original, Zraw[subject_index]]) (NOTE input order differs from W)",
            "decoder_source": "exact frozen S3 DiscreteNeuralProcess.decoder",
            "decoder_device": str(device_obj),
            "decoder_sampling": "none; eval mode, original raw S5 posterior means",
            "decoder_choice": "argmax of two class logits, ties mapped to class 0",
            "logit_nn": "binary log-odds = raw_class_logit_1 - raw_class_logit_0",
            "p_nn": "P(choice=1) = softmax(two class logits)[1]",
            "choice_nn": "0/1 deterministic decoder argmax (not Bernoulli draw)",
            "choice_observed": "actual 0/1 response from original S1 dataset",
            "labels_leaked_to_z": True,
            "leakage_note": "All-observed-trial S5 posterior means were inferred using the same original choices. Descriptive in-sample oracle; NOT independent held-out target prediction.",
            "source_dataset": str(dataset_path),
            "source_dataset_sha256": file_sha256(dataset_path),
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": file_sha256(checkpoint_path),
            "source_s5": source_provenance or {},
            "files": {
                "Zstd.npy": [n, dz], "Zraw.npy": [n, dz],
                "R.npy": [dz, dz], "W.npy": [m, dz + dx],
                "Y.npz": {
                    "choice_observed": [m], "logit_nn": [m],
                    "p_nn": [m], "choice_nn": [m],
                    "logits_nn_2class": [m, 2]},
                "subject_index.npy": [m], "trial_index.npy": [m],
                "trial_counts.npy": [n],
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2)+"\n",
            encoding="utf-8")

        # Build complete data in a staging folder and swap only on success.
        # Restore an earlier export if the final rename fails.
        backup = None
        if target.exists():
            backup = out / (".stage1_data_backup_" +
                            hashlib.sha256(str(checkpoint_path).encode()).hexdigest()[:12])
            if backup.exists():
                raise FileExistsError(f"Cannot replace export: stale backup exists: {backup}")
            os.replace(target, backup)
        try:
            os.replace(stage, target)
        except BaseException:
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)
            raise
        if backup is not None:
            shutil.rmtree(backup)
    return {"directory": str(target), "n_subjects": int(n),
            "n_trials_total": m, "z_dim": dz, "x_dim": dx,
            "manifest": str(target / "manifest.json")}
