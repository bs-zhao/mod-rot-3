"""Verified access to previously fitted WY-only response-bank functions."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import run as wave


class FittedBank:
    def __init__(self, source, folder):
        self.folder = Path(folder)
        self.derived_reads = set()
        self.protocol = json.loads(self.read("protocol.json").decode())
        if self.protocol["status"] != "COMPLETE":
            raise ValueError("Incomplete source function fit")
        if "fit_manifest_sha256" in self.protocol:
            manifest_bytes = self.read("fit_manifest.json")
            self.source_digest = hashlib.sha256(manifest_bytes).hexdigest()
            if self.source_digest != self.protocol["fit_manifest_sha256"]:
                raise ValueError("Source function manifest changed")
            for name, expected in json.loads(manifest_bytes).items():
                if Path(name).name != name or hashlib.sha256(self.read(name)).hexdigest() != expected:
                    raise ValueError("Source function artifact changed")
        else:
            self.source_digest = hashlib.sha256(self.read("blind_scores.csv")).hexdigest()
            if self.source_digest != self.protocol["scores_sha256"]:
                raise ValueError("Source frozen scores changed")
        self.data = wave.io.MatrixCases(source, tuple(self.protocol["channels"]), self.protocol["seed"])
        if self.data.index["model"] != self.protocol["model"] or self.data.n != self.protocol["n_subjects"]:
            raise ValueError("Source functions do not match W data")
        rng = np.random.default_rng(self.protocol["seed"])
        self.train = np.sort(rng.permutation(self.data.n)[:int(.75*self.data.n)])
        mean = self.data.u[self.train].mean(0)
        scale = np.sqrt(np.var(self.data.u[self.train], axis=0).mean())
        self.u = (self.data.u-mean)/scale
        self.derived_reads.add("response_bank.npz")
        with np.load(self.folder / "response_bank.npz", allow_pickle=False) as saved:
            self.bank = saved["responses"].copy()
        self.variance = np.maximum(self.bank[self.train].var(axis=0), 1e-12)
        self.derived_reads.add("fit_quality.csv")
        with (self.folder / "fit_quality.csv").open(newline="") as stream:
            self.quality = {(r["estimator"], r["channel"]): r for r in csv.DictReader(stream)}

    def read(self, name):
        self.derived_reads.add(name)
        return (self.folder / name).read_bytes()

    def model(self, name):
        filename = f"model_{name}.npz"
        self.derived_reads.add(filename)
        with np.load(self.folder / filename, allow_pickle=False) as saved:
            coef = saved["coef"].copy()
            scaler = object.__new__(wave.sur.Scaler)
            scaler.mean, scaler.scale = saved["mean"].copy(), saved["scale"].copy()
            center_points, stimuli = saved["centers"].copy(), saved["score_stimuli"].copy()
        result = dict(z=scaler(self.u), scaler=scaler)
        if name.startswith("poly"):
            powers = wave.sur.powers_for(self.data.d, int(name[4:]))
            if len(powers) != len(coef):
                raise ValueError("Polynomial coefficient dimensions changed")
            result.update(powers=powers, weights=wave.sur.weights_for(powers))
            coefficients = coef
        else:
            model = wave.sur.RBF(scaler, center_points, float(name[3:]))
            model.set_coef(coef)
            field_name = f"field_{name}.npz"
            self.derived_reads.add(field_name)
            with np.load(self.folder / field_name, allow_pickle=False) as saved:
                subjects, expected = saved["subjects"][:8], saved["gradient"][:8]
            actual, _ = model.derivatives(self.u[subjects])
            ids = np.r_[stimuli, self.bank.shape[2]+stimuli]
            actual = actual[:, ids].reshape(expected.shape)
            if np.max(np.abs(actual-expected)) > 1e-6*max(1., np.max(np.abs(expected))):
                raise ArithmeticError("Reconstructed kernel disagrees with saved fields")
            result.update(centers=model.centers, lengthscale=model.lengthscale)
            coefficients = model.beta
        grams = []
        for channel in range(len(self.data.channels)):
            values = coefficients[:, channel*self.bank.shape[2]:(channel+1)*self.bank.shape[2]]
            normalized = values/np.sqrt(self.variance[channel])[None, :]
            grams.append(normalized @ normalized.T)
        result["grams"] = grams
        return result
