"""CR4: three-parameter CPT with alpha FIXED to 1.

Exactly the original CR2/CR3 lottery valuation, loss weighting, choice rule
and lambda/gamma/tau population priors; removes only alpha. Independent,
continuous, valid-domain mixed-lottery X: NO zero amounts, equal-probability
condition, gain-only/loss-only conditions, FOSD rejection, or Fisher design.
All subjects share a deterministic 500-trial bank. Previous data untouched.

From repository root:
    python np/reference/cr4/s0_s1_cr4.py --stage design
    python np/reference/cr4/s0_s1_cr4.py --stage dataset --subjects 5000
    python np/reference/cr4/s0_s1_cr4.py --stage selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(RESEARCH), str(ROOT / "np" / "src")]
from s0_cr2_design_recovery import sample_prior, logits as cr2_logits
from modlearn_rot.generators.dataset import GeneratedDataset, save_dataset

OUT = ROOT / "outputs" / "choice" / "CR4"
BANK = OUT / "s0_design" / "CR4_3P_random500.npz"
DATASET = OUT / "s1_generate" / "s1_generate__dataset.npz"
PARAMETER_NAMES = ("lambda_loss", "gamma", "tau")
X_NAMES = ("gain_A", "loss_A", "p_gain_A", "gain_B", "loss_B", "p_gain_B")
N_TRIALS = 500
DEFAULT_AMOUNT_RANGE = (5.0, 200.0)
DEFAULT_PROB_RANGE = (0.05, 0.95)


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for data in iter(lambda: file.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sample_theta(n, rng):
    """Keep the CR3 lambda/gamma/tau marginal priors, drop only alpha."""
    return np.asarray(sample_prior(n, 4, rng)[:, 1:], dtype=np.float64)


def theta_as_cr3(theta):
    """Insert fixed alpha=1 without touching the CR2/CR3 mathematical code."""
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape[-1] != 3:
        raise ValueError("CR4 theta must have (lambda_loss,gamma,tau)")
    return np.concatenate((np.ones_like(theta[..., :1]), theta), axis=-1)


def logits(x, theta):
    return cr2_logits(np.asarray(x, dtype=np.float64), theta_as_cr3(theta))


def sample_bank(seed=20261023, n=N_TRIALS,
                amount_range=DEFAULT_AMOUNT_RANGE,
                prob_range=DEFAULT_PROB_RANGE):
    """Each of the six X attributes is INDEPENDENTLY uniform continuous.

    Boundaries are excluded almost surely; there is deliberately no rejection
    or design optimization, even if a lottery happens to dominate another.
    """
    alo, ahi = map(float, amount_range)
    plo, phi = map(float, prob_range)
    if not (0 < alo < ahi and 0 < plo < phi < 1):
        raise ValueError("Need 0 < amount_min < amount_max and 0 < p_min < p_max < 1")
    if n < 1:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    x = np.empty((n, 6), dtype=np.float64)
    x[:, [0, 1, 3, 4]] = rng.uniform(alo, ahi, size=(n, 4))
    x[:, [2, 5]] = rng.uniform(plo, phi, size=(n, 2))
    return x


def verify_bank(x, amount_range=DEFAULT_AMOUNT_RANGE,
                prob_range=DEFAULT_PROB_RANGE):
    x = np.asarray(x)
    alo, ahi = amount_range
    plo, phi = prob_range
    return (x.shape == (N_TRIALS, 6)
            and np.isfinite(x).all()
            and np.all((x[:, [0, 1, 3, 4]] >= alo)
                       & (x[:, [0, 1, 3, 4]] <= ahi))
            and np.all((x[:, [2, 5]] >= plo)
                       & (x[:, [2, 5]] <= phi))
            and np.all(x[:, [0, 1, 3, 4]] > 0)
            and np.all(x[:, [2, 5]] > 0)
            and np.all(x[:, [2, 5]] < 1))


def design(args):
    if BANK.exists():
        raise FileExistsError(f"Existing CR4 bank preserved: {BANK}")
    if DATASET.exists():
        raise FileExistsError("CR4 dataset exists: never replace its stimulus bank")
    x = sample_bank(args.seed, amount_range=(args.amount_min, args.amount_max),
                    prob_range=(args.prob_min, args.prob_max))
    if not verify_bank(x, (args.amount_min, args.amount_max),
                       (args.prob_min, args.prob_max)):
        raise AssertionError("CR4 random-bank constraints failed")
    BANK.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BANK, X=x, parameter_names=np.asarray(PARAMETER_NAMES),
                        x_names=np.asarray(X_NAMES))
    write_json(BANK.parent / "CR4_3P_design.json", dict(
        model="CR4", fixed_alpha=1.0, parameter_names=list(PARAMETER_NAMES),
        trials=N_TRIALS, seed=args.seed,
        x_sampling="six independent continuous uniform coordinates; shared 500-item bank",
        amount_range=[args.amount_min, args.amount_max],
        probability_range=[args.prob_min, args.prob_max],
        excluded_special_conditions=True, excluded_boundary_values=True,
        independent_X=True, no_FOSD_rejection=True, no_Fisher_optimization=True,
        bank_sha256=hash_file(BANK)))
    print("CR4 independent continuous 500-X bank:", BANK, flush=True)


def dataset(args):
    if DATASET.exists():
        raise FileExistsError(f"Existing CR4 dataset preserved: {DATASET}")
    if not BANK.exists():
        raise FileNotFoundError(f"Generate CR4 bank first: {BANK}")
    with np.load(BANK, allow_pickle=False) as saved:
        x = np.asarray(saved["X"], dtype=np.float64)
    design_meta = json.loads((BANK.parent / "CR4_3P_design.json").read_text(encoding="utf-8"))
    if not verify_bank(x, design_meta["amount_range"], design_meta["probability_range"]):
        raise ValueError("CR4 bank invalid or modified")
    rng_theta = np.random.default_rng(args.data_seed + 4)
    rng_choice = np.random.default_rng(args.data_seed + 1004)
    theta = sample_theta(args.subjects, rng_theta)
    p = expit(logits(x[None, :, :], theta[:, None, :]))
    choices = (rng_choice.random(p.shape) < p).astype(np.int64)
    if not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
        raise AssertionError("Nonfinite or invalid CR4 Bernoulli probability")
    n = len(theta)
    data = GeneratedDataset(
        parameters=theta,
        trial_counts=np.full(n, N_TRIALS, dtype=np.int64),
        subject_index=np.repeat(np.arange(n, dtype=np.int64), N_TRIALS),
        trial_index=np.tile(np.arange(N_TRIALS, dtype=np.int64), n),
        x=np.broadcast_to(x, (n, N_TRIALS, 6)).reshape(-1, 6).copy(),
        y=choices.reshape(-1, 1),
        predictive_true=p.reshape(-1, 1),
    )
    save_dataset(data, DATASET)
    write_json(DATASET.parent / "s1_generate__manifest.json", dict(
        model="CR4", n_parameters=3, fixed_alpha=1.0, subjects=n,
        trials_per_subject=N_TRIALS, x_dim=6,
        parameter_names=list(PARAMETER_NAMES),
        data_seed=args.data_seed, design_seed=design_meta["seed"],
        bank_sha256=hash_file(BANK), dataset_sha256=hash_file(DATASET),
        formula="CR2/CR3 logits with alpha=1 exactly",
        lambda_gamma_tau_priors="unchanged CR2/CR3 marginals",
        x_sampling="independent continuous mixed lottery; no special trial classes",
        decoder_x_preprocessing="z-score CR4 bank X in np/src/modrot3/pipeline.py",
    ))
    print(f"Generated CR4 {n} subjects x {N_TRIALS} trials:", DATASET, flush=True)


def selftest():
    a = sample_bank(17)
    b = sample_bank(17)
    assert np.array_equal(a, b) and verify_bank(a)
    theta = sample_theta(8, np.random.default_rng(19))
    assert theta.shape == (8, 3)
    expected = cr2_logits(a[None], theta_as_cr3(theta)[:, None])
    assert np.allclose(logits(a[None], theta[:, None]), expected)
    assert np.isfinite(expected).all()
    print("CR4 no-special-trial bank, priors and CR3-formula equivalence PASS")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--stage", choices=("design", "dataset", "selftest"), default="design")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--data-seed", type=int, default=20261028)
    parser.add_argument("--subjects", type=int, default=5000)
    parser.add_argument("--amount-min", type=float, default=DEFAULT_AMOUNT_RANGE[0])
    parser.add_argument("--amount-max", type=float, default=DEFAULT_AMOUNT_RANGE[1])
    parser.add_argument("--prob-min", type=float, default=DEFAULT_PROB_RANGE[0])
    parser.add_argument("--prob-max", type=float, default=DEFAULT_PROB_RANGE[1])
    args = parser.parse_args()
    if args.subjects < 1:
        parser.error("subjects must be positive")
    if args.stage == "selftest":
        selftest()
    elif args.stage == "design":
        design(args)
    else:
        dataset(args)


if __name__ == "__main__":
    main()
