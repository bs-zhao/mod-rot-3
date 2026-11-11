"""Prepare immutable WY-fitted function banks without redundant case scoring."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

import run as wave
import run_finite


def prepare(source, output, estimators=("poly3",), seed=383, centers=512):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Retain existing bank; choose a new folder: {output}")
    data = wave.io.MatrixCases(source, ("logit_nn", "p_nn"), seed)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=data.index["model"], Z_dim=data.d, n_subjects=data.n,
        source=str(source.resolve()), seed=seed, channels=list(data.channels), estimators=list(estimators),
        centers=centers, heldout_fraction=.25, ridge_grid=[1e-6, 1e-4, .01],
        all_subjects_used=True, artifact_type="WY-fitted function preparation; no case scores",
        status="RUNNING", code_sha256=hashlib.sha256(b"".join((wave.HERE / f).read_bytes() for f in
            ("prepare_response_bank.py", "run_finite.py", "additive_profiles.py", "matrix_io.py", "surrogates.py"))).hexdigest())
    wave.json_file(output / "protocol.json", protocol)
    original = source.parent / "step5_matrix_metrics" / "waveC_finite_seed383"
    reuse = False
    if (original / "protocol.json").exists():
        old = json.loads((original / "protocol.json").read_text())
        expected_code = hashlib.sha256(b"".join((wave.HERE / f).read_bytes() for f in
            ("run_finite.py", "finite_effects.py", "additive_profiles.py", "matrix_io.py", "surrogates.py", "structure.py"))).hexdigest()
        reuse = (old["status"] == "COMPLETE" and old["seed"] == seed and old["centers"] == centers and
                 old["model"] == data.index["model"] and old["channels"] == list(data.channels) and
                 old["code_sha256"] == expected_code and old["source"] == str(source.resolve()) and
                 old["n_subjects"] == data.n and set(estimators) <= set(old["estimators"]) and
                 hashlib.sha256((original / "blind_scores.csv").read_bytes()).hexdigest() == old["scores_sha256"])
    quality = []; artifacts = ["response_bank.npz", "response_quality.json", "fit_quality.csv"]
    if reuse:
        for name in ("response_bank.npz", "response_quality.json"):
            shutil.copyfile(original / name, output / name)
        with (original / "fit_quality.csv").open(newline="") as stream:
            quality = [r for r in csv.DictReader(stream) if r["estimator"] in estimators]
        for name in estimators:
            files = [f"model_{name}.npz", f"fit_{name}.json"]
            if name.startswith("rbf"):
                files.append(f"field_{name}.npz")
            for filename in files:
                shutil.copyfile(original / filename, output / filename)
                artifacts.append(filename)
        audit = dict(reused=True, source=str(original.resolve()), source_code_sha256=old["code_sha256"],
                     source_scores_sha256=old["scores_sha256"])
    else:
        rng = np.random.default_rng(seed)
        train = np.sort(rng.permutation(data.n)[:int(.75*data.n)])
        mean = data.u[train].mean(0)
        scale = np.sqrt(np.var(data.u[train], axis=0).mean())
        u = (data.u-mean)/scale
        bank, x, response_quality = run_finite.response_bank(data, seed)
        wave.json_file(output / "response_quality.json", response_quality)
        stim = np.sort(rng.choice(bank.shape[2], min(64, bank.shape[2]), replace=False))
        ids = np.r_[stim, bank.shape[2]+stim]
        center_ids = rng.choice(train, min(centers, len(train)), replace=False)
        np.savez_compressed(output / "response_bank.npz", responses=bank, x=x, score_stimuli=stim)
        target = bank.reshape(data.n, -1)
        scaler = wave.sur.Scaler(u, data.d, row_mask=np.isin(np.arange(data.n), train))
        for name in estimators:
            model = (wave.sur.Polynomial(scaler, int(name[4:])) if name.startswith("poly") else
                     wave.sur.RBF(scaler, u[center_ids], float(name[3:])))
            fit = wave.sur.fit_all_rows(model, u, target, np.arange(data.n), train, chunk=1024)
            wave.json_file(output / f"fit_{name}.json", fit)
            np.savez_compressed(output / f"model_{name}.npz", coef=model.coef, mean=scaler.mean,
                               scale=scaler.scale, centers=u[center_ids], score_stimuli=stim)
            artifacts.extend((f"fit_{name}.json", f"model_{name}.npz"))
            nmse = np.asarray(fit["validation_nmse"]).reshape(2, bank.shape[2]).mean(1)
            for c, channel in enumerate(data.channels):
                quality.append(dict(estimator=name, channel=channel, validation_nmse=float(nmse[c]),
                    profile_nmse=response_quality["validation_nmse"][c],
                    adequate=bool(nmse[c] <= .1 and response_quality["validation_nmse"][c] <= .1),
                    final_fit_subjects=data.n, final_fit_trial_rows=data.m))
            # A small derivative verification record is sufficient for the
            # loader; the bank is not a full finite-effect experiment.
            if name.startswith("rbf"):
                model.set_coef(model.coef[:, ids])
                g, _ = model.derivatives(u[:8])
                np.savez_compressed(output / f"field_{name}.npz", subjects=np.arange(8),
                                    gradient=g.reshape(8, 2, len(stim), data.d))
                artifacts.append(f"field_{name}.npz")
            print(f"[Prepare] {data.index['model']} {name}: all-subject function fit saved", flush=True)
        audit = dict(reused=False, files=sorted(set(data.read_files)))
    wave.csv_file(output / "fit_quality.csv", quality)
    manifest = {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in artifacts}
    wave.json_file(output / "fit_manifest.json", manifest)
    wave.json_file(output / "preparation_audit.json", dict(**audit, forbidden_files_read=[],
        all_subjects=data.n, all_rows=data.m, case_scores_computed=False))
    protocol.update(status="COMPLETE", fit_manifest_sha256=hashlib.sha256((output / "fit_manifest.json").read_bytes()).hexdigest())
    wave.json_file(output / "protocol.json", protocol)
    return protocol
