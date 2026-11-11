"""Immutable joint-projection coefficients and actual-data Gram matrices."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import run as wave
import fast_projection as fp


def prepare(source, folder, seed=391, degrees=(2, 3, 4), chunk=8192, device="auto"):
    source, folder = Path(source), Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError(f"Retain existing bank: {folder}")
    data = wave.io.MatrixCases(source, ("logit_nn", "p_nn"), seed)
    folder.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=data.index["model"], Z_dim=data.d, n_subjects=data.n,
        source=str(source.resolve()), seed=seed, degrees=list(degrees), channels=list(data.channels),
        chunk=chunk, device=device, heldout_fraction=.25, ridge_grid=[1e-6, 1e-4, .01],
        all_subjects_used=True, artifact_type="joint projection and empirical Gram preparation; no case scores",
        status="RUNNING", code_sha256=hashlib.sha256(b"".join((wave.HERE / f).read_bytes() for f in
            ("joint_bank.py", "fast_projection.py", "matrix_io.py", "surrogates.py"))).hexdigest())
    wave.json_file(folder / "protocol.json", protocol)
    rng = np.random.default_rng(seed)
    train = np.sort(rng.permutation(data.n)[:int(.75*data.n)])
    scaler = wave.sur.Scaler(data.w, data.d, row_mask=np.isin(data.sid, train))
    quality, files = [], ["fit_quality.csv"]
    for degree in degrees:
        model = fp.Polynomial(scaler, degree)
        stats_name = f"statistics_{model.name}.npz"
        coef, fit = fp.fit(model, data.w, data.y, data.sid, train, chunk=chunk,
                           device=device, statistics_output=folder / stats_name)
        projection_name = f"projection_{model.name}.npz"
        np.savez_compressed(folder / projection_name, coef=coef, powers=model.powers, weights=model.weights,
                            mean=scaler.mean, scale=scaler.scale, train_subjects=train)
        wave.json_file(folder / f"fit_{model.name}.json", fit)
        files.extend((stats_name, projection_name, f"fit_{model.name}.json"))
        protocol["numeric_backend"] = fit["backend"]
        for c, channel in enumerate(data.channels):
            quality.append(dict(estimator=model.name, channel=channel, validation_nmse=fit["validation_nmse"][c],
                                adequate=fit["validation_nmse"][c] <= .1,
                                final_fit_subjects=data.n, final_fit_rows=data.m))
        print(f"[Joint bank] {data.index['model']} {model.name}: all original rows saved", flush=True)
    wave.csv_file(folder / "fit_quality.csv", quality)
    wave.json_file(folder / "manifest.json", {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in files})
    wave.json_file(folder / "read_audit.json", dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
                   all_subjects=data.n, all_trial_rows=data.m, case_scores_computed=False))
    protocol.update(status="COMPLETE", manifest_sha256=hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest())
    wave.json_file(folder / "protocol.json", protocol)
    return protocol


class JointBank:
    def __init__(self, source, folder):
        self.folder = Path(folder)
        self.protocol = json.loads((self.folder / "protocol.json").read_text())
        raw = (self.folder / "manifest.json").read_bytes()
        if self.protocol["status"] != "COMPLETE" or hashlib.sha256(raw).hexdigest() != self.protocol["manifest_sha256"]:
            raise ValueError("Incomplete or changed joint function bank")
        for name, digest in json.loads(raw).items():
            if Path(name).name != name or hashlib.sha256((self.folder/name).read_bytes()).hexdigest() != digest:
                raise ValueError("Joint projection artifact changed")
        self.data = wave.io.MatrixCases(source, tuple(self.protocol["channels"]), self.protocol["seed"])
        if (self.data.index["model"] != self.protocol["model"] or self.data.n != self.protocol["n_subjects"] or
                str(Path(source).resolve()) != self.protocol["source"]):
            raise ValueError("Joint bank does not match selected case matrices")
        with (self.folder / "fit_quality.csv").open(newline="") as stream:
            self.quality = {(r["estimator"], r["channel"]): r for r in csv.DictReader(stream)}

    def load(self, degree):
        with np.load(self.folder / f"projection_poly{degree}.npz", allow_pickle=False) as saved:
            out = {key: saved[key].copy() for key in saved.files}
        with np.load(self.folder / f"statistics_poly{degree}.npz", allow_pickle=False) as saved:
            out.update({key: saved[key].copy() for key in saved.files})
        return out
