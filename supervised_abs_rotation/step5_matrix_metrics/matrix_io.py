"""Only W, common Y, and row-layout metadata may enter this experiment."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

CHANNELS = ("logit_nn", "p_nn", "choice_nn", "choice_observed")
DIMS = {"C1": 4, "C2": 4, "C3": 6, "CR3": 4}


def default_step2(root, model):
    beta = "5em05" if model == "CR3" else "0p0005"
    return (Path(root) / "outputs" / "supervised_abs_rotation" / model /
            f"beta{beta}_lambda0_ctx0p25to0p8_L{DIMS[model]}" /
            "repeat01" / "abs_mse" / "step2_pairwise_sweep")


class MatrixCases:
    """Read anonymous coordinates from actual case W, never saved R or theta."""

    def __init__(self, folder, channels, seed=381):
        self.folder = Path(folder)
        self.index = json.loads((self.folder / "index.json").read_text(encoding="utf-8"))
        self.d = int(self.index["Z_dim"])
        self.dx = int(self.index["X_dim"])
        self.n = int(self.index["n_subjects"])
        self.m = int(self.index["n_trials"])
        if not self.index.get("complete"):
            raise ValueError("Step 2 is incomplete")
        self.paths = sorted((self.folder / "cases").glob("*/W.npy"))
        if len(self.paths) != 1 + 6 * math.comb(self.d, 2):
            raise ValueError("Not all pair-angle W matrices exist")
        if {p.parent.name for p in self.paths} != {c["id"] for c in self.index["cases"]}:
            raise ValueError("Case index does not match W files")
        self.sid = np.load(self.folder / "subject_index.npy", allow_pickle=False)
        counts = np.load(self.folder / "trial_counts.npy", allow_pickle=False)
        self.offsets = np.r_[0, np.cumsum(counts)]
        if (counts.shape != (self.n,) or np.any(counts <= 0) or
                self.offsets[-1] != self.m or
                not np.array_equal(self.sid, np.repeat(np.arange(self.n), counts))):
            raise ValueError("Invalid subject row mapping")
        tid = np.load(self.folder / "trial_index.npy", allow_pickle=False)
        expected_tid = np.arange(self.m) - np.repeat(self.offsets[:-1], counts)
        if not np.array_equal(tid, expected_tid):
            raise ValueError("Invalid trial row mapping")
        if not channels or not set(channels) <= set(CHANNELS):
            raise ValueError("Unknown Y channel")
        self.channels = tuple(channels)
        with np.load(self.folder / "Y.npz", allow_pickle=False) as ys:
            self.y = np.column_stack([ys[k] for k in channels]).astype(np.float64)
        if self.y.shape != (self.m, len(channels)) or not np.isfinite(self.y).all():
            raise ValueError("Y does not match W rows")
        rng = np.random.default_rng(seed)
        self.paths = [self.paths[i] for i in rng.permutation(len(self.paths))]
        self.w = np.load(self.paths[0], mmap_mode="r", allow_pickle=False)
        if self.w.shape != (self.m, self.d + self.dx):
            raise ValueError("Wrong W shape")
        self.u = np.asarray(self.w[self.offsets[:-1], :self.d], dtype=float)
        self.read_files = ["index.json", "subject_index.npy", "trial_counts.npy",
                           "trial_index.npy", "Y.npz"]

    def coordinates(self, path, chunk=65536):
        w = np.load(path, mmap_mode="r", allow_pickle=False)
        if w.shape != self.w.shape:
            raise ValueError(f"Wrong W shape: {path}")
        u = np.asarray(w[self.offsets[:-1], :self.d], dtype=float)
        for a in range(0, self.m, chunk):
            b = min(a + chunk, self.m)
            if (not np.isfinite(w[a:b]).all() or
                    not np.allclose(w[a:b, :self.d], u[self.sid[a:b]], atol=1e-6) or
                    not np.array_equal(w[a:b, self.d:], self.w[a:b, self.d:])):
                raise ValueError(f"Inconsistent coordinates or X in {path}")
        self.read_files.append(str(path.relative_to(self.folder)))
        return u

    def evaluation_rows(self, per_subject=8):
        """Every subject contributes equally, at observed W locations only."""
        rows = []
        for a, b in zip(self.offsets[:-1], self.offsets[1:]):
            rows.extend(np.linspace(a, b - 1, min(per_subject, b - a), dtype=int))
        return np.asarray(rows, dtype=int)


def registration(reference, candidate):
    """Q satisfies U_candidate = U_reference @ Q; no angle metadata used."""
    a = reference - reference.mean(0)
    b = candidate - candidate.mean(0)
    left, _, right = np.linalg.svd(a.T @ b, full_matrices=False)
    q = left @ right
    relative_error = np.linalg.norm(reference @ q - candidate) / max(np.linalg.norm(candidate), 1e-15)
    if relative_error > 2e-5:
        raise ValueError("Cases are not orthogonal rotations of the same U")
    return q, float(relative_error)
