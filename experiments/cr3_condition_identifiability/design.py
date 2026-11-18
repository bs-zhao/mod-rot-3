"""Load exact CR3 bank and define condition-valid perturbations.

Every condition label is loaded from the original S0 bank; the four groups
are mutually exclusive. Zero is a meaningful designed amount, not missing
data. X stays in physical units for boundary rules.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / "outputs/choice/CR3/s0_design/CR3_4P_fixed500.npz"
DATASET = ROOT / "outputs/choice/CR3/s1_generate/s1_generate__dataset.npz"
NAMES = ("gain_only", "gamma_neutral", "loss_only", "general_CR2")
PARAM_NAMES = ("alpha", "lambda", "gamma", "tau")
X_NAMES = ("gain_A", "loss_A", "p_A", "gain_B", "loss_B", "p_B")
EXPECTED = {"gain_only": 60, "gamma_neutral": 60, "loss_only": 40, "general_CR2": 340}
AMOUNT_AXES = (0, 1, 3, 4)
PROB_AXES = (2, 5)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class Design:
    x: np.ndarray
    codes: np.ndarray
    theta: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    x_sd: np.ndarray
    subject_ids: np.ndarray
    trial_ids: np.ndarray
    bank_hash: str
    dataset_hash: str

    @property
    def t(self):
        return (self.theta - self.mu) / self.sd


def condition_mask(x, codes):
    """Check bank's declared categories rather than inferring overlapping types."""
    x = np.asarray(x)
    codes = np.asarray(codes)
    assert x.shape == (len(codes), 6)
    if not np.isin(codes, np.arange(4)).all():
        raise ValueError("unexpected condition code")
    gain = np.isclose(x[:, 1], 0) & np.isclose(x[:, 4], 0)
    loss = np.isclose(x[:, 0], 0) & np.isclose(x[:, 3], 0)
    neutral = np.isclose(x[:, 2], .5) & np.isclose(x[:, 5], .5)
    if not np.all(gain[codes == 0]):
        raise ValueError("gain-only rows do not have zero loss")
    if not np.all(neutral[codes == 1]):
        raise ValueError("gamma-neutral rows do not have equal 0.5 probabilities")
    if not np.all(loss[codes == 2]):
        raise ValueError("loss-only rows do not have zero gains")
    if np.any((gain | loss | neutral)[codes == 3]):
        raise ValueError("general trial assigned a special type")
    return {name: codes == i for i, name in enumerate(NAMES)}


def subset_stratified(codes, n_stimuli, rng):
    if n_stimuli <= 0 or n_stimuli >= len(codes):
        return np.arange(len(codes), dtype=int)
    if n_stimuli < 4:
        raise ValueError("at least 4 stimuli needed for all four conditions")
    counts = np.bincount(codes, minlength=4)
    # Exactly one guaranteed per condition; remaining apportioned near
    # empirical proportions while respecting original available counts.
    selected = [int(rng.choice(np.flatnonzero(codes == s))) for s in range(4)]
    remaining = n_stimuli - 4
    while remaining:
        remaining_ids = np.setdiff1d(np.arange(len(codes)), selected)
        s = int(rng.choice(4, p=np.bincount(codes[remaining_ids], minlength=4) /
                           len(remaining_ids)))
        selected.append(int(rng.choice(remaining_ids[codes[remaining_ids] == s])))
        remaining -= 1
    return np.sort(np.asarray(selected, dtype=int))


def load_design(subjects=32, stimuli=500, seed=20261117,
                bank=BANK, dataset=DATASET):
    bank, dataset = Path(bank), Path(dataset)
    if not bank.exists() or not dataset.exists():
        raise FileNotFoundError(
            "CR3 bank and S1 dataset are required: " + str(bank) + " and " + str(dataset)
        )
    with np.load(bank, allow_pickle=False) as b:
        full_x = np.asarray(b["X"], dtype=np.float64)
        codes = np.asarray(b["type_codes"], dtype=int)
        names = tuple(str(z) for z in b["type_names"])
    if full_x.shape != (500, 6) or names != NAMES:
        raise ValueError("CR3 fixed bank has changed; expected 500 X rows and exact four type names")
    condition_mask(full_x, codes)
    full_counts = {name: int((codes == i).sum()) for i, name in enumerate(NAMES)}
    if full_counts != EXPECTED:
        raise ValueError(f"Unexpected CR3 condition counts: {full_counts}")
    with np.load(dataset, allow_pickle=False) as d:
        all_theta = np.asarray(d["parameters"], dtype=np.float64)
    if all_theta.ndim != 2 or all_theta.shape[1] != 4:
        raise ValueError("CR3 dataset must supply (n_subjects,4) true parameters")
    mu, sd = all_theta.mean(axis=0), all_theta.std(axis=0)
    if np.any(sd <= 0):
        raise ValueError("degenerate theta standardization")
    rng = np.random.default_rng(seed)
    sid = np.sort(rng.choice(len(all_theta), min(subjects, len(all_theta)), replace=False))
    tid = subset_stratified(codes, stimuli, rng)
    x = full_x[tid]
    return Design(x=x, codes=codes[tid], theta=all_theta[sid],
                  mu=mu, sd=sd, x_sd=np.maximum(full_x.std(axis=0), 1e-12),
                  subject_ids=sid, trial_ids=tid,
                  bank_hash=sha256(bank), dataset_hash=sha256(dataset))


def derivative_masks(x, codes):
    """Two meanings of X derivative: ambient interior versus condition tangent.

    On gamma-neutral trials p=.5 is ambient-smooth, but changing p moves
    outside the gamma-neutral condition; exclude its p axes from tangent.
    Zero amounts are intrinsically singular as X variables and excluded
    in both masks (even if torch.where's branch Jacobian returns zero).
    """
    x = np.asarray(x, dtype=np.float64)
    valid = np.zeros_like(x, dtype=bool)
    valid[:, list(AMOUNT_AXES)] = x[:, list(AMOUNT_AXES)] > 0
    valid[:, list(PROB_AXES)] = ((x[:, list(PROB_AXES)] > 0) &
                                (x[:, list(PROB_AXES)] < 1))
    tangent = valid.copy()
    tangent[codes == 0, 1] = False
    tangent[codes == 0, 4] = False
    tangent[codes == 2, 0] = False
    tangent[codes == 2, 3] = False
    tangent[codes == 1, 2] = False
    tangent[codes == 1, 5] = False
    return valid, tangent


def physical_finite_probes(x, codes, amount_step=5.0, probability_step=.05):
    """Legal one-sided or centered finite intervention, NEVER boundary AD.

    Returns (plus, minus, widths, flags, masks). flags:
      0 = within-condition tangent (may be backward at empirical upper edge)
      1 = zero / endpoint boundary; one-sided counterfactual
      2 = smooth ambient but leaves a fixed-design condition
    No 'finite derivative at 0' is claimed. All values stay within the
    empirical support [0,200] or probability [0,1].
    """
    if amount_step <= 0 or not (0 < probability_step < .5):
        raise ValueError("positive steps required, probability_step < 0.5")
    x = np.asarray(x, dtype=np.float64)
    valid, tangent = derivative_masks(x, codes)
    steps = np.array([amount_step, amount_step, probability_step,
                      amount_step, amount_step, probability_step], dtype=float)
    lower = np.zeros(6)
    upper = np.array([200, 200, 1, 200, 200, 1], dtype=float)
    k = len(x)
    xp = np.repeat(x[:, None, :], 6, axis=1)
    xm = xp.copy()
    widths = np.empty((k, 6), dtype=float)
    flags = np.full((k, 6), 2, dtype=np.int8)
    onesided = np.zeros((k, 6), dtype=bool)
    for i in range(k):
        for j in range(6):
            delta = steps[j]
            a = x[i, j]
            if a - delta >= lower[j] - 1e-12 and a + delta <= upper[j] + 1e-12:
                xp[i, j, j] = a + delta
                xm[i, j, j] = a - delta
                widths[i, j] = 2 * delta
            elif a + delta <= upper[j] + 1e-12:
                xp[i, j, j] = a + delta
                widths[i, j] = delta
                onesided[i, j] = True
            elif a - delta >= lower[j] - 1e-12:
                xm[i, j, j] = a - delta
                widths[i, j] = delta
                onesided[i, j] = True
            else:
                raise ValueError(f"no valid contrast at row {i}, axis {j}")
            if tangent[i, j]:
                flags[i, j] = 0
            elif not valid[i, j]:
                flags[i, j] = 1
    assert (xp >= lower - 1e-10).all() and (xm >= lower - 1e-10).all()
    assert (xp <= upper + 1e-10).all() and (xm <= upper + 1e-10).all()
    return xp, xm, widths, flags, valid, tangent, onesided
