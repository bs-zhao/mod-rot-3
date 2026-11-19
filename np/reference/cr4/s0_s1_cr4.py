"""CR4 3P CPT (alpha=1), with independent X for every subject and trial.

X is sampled independently of T and choices. Neither CR3 nor CR5's fixed
stimulus banks are changed.
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
SAMPLING_POLICY = "iid_subject_trial_uniform_v1"


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


def sample_subject_x(subjects, seed, trials=N_TRIALS,
                     amount_range=DEFAULT_AMOUNT_RANGE,
                     prob_range=DEFAULT_PROB_RANGE):
    """Generate IID X for each (subject, trial), shape (subjects,trials,6)."""
    if subjects < 1 or trials < 1:
        raise ValueError("subjects and trials must be positive")
    return sample_bank(seed=seed, n=subjects * trials,
                       amount_range=amount_range,
                       prob_range=prob_range).reshape(subjects, trials, 6)


def verify_x(x, amount_range=DEFAULT_AMOUNT_RANGE,
             prob_range=DEFAULT_PROB_RANGE):
    """Verify physical support without assuming a shared 500-row bank."""
    x = np.asarray(x)
    if x.ndim < 2 or x.shape[-1] != 6 or not np.isfinite(x).all():
        return False
    lo, hi = amount_range
    plo, phi = prob_range
    amt = x[..., [0, 1, 3, 4]]
    prob = x[..., [2, 5]]
    return bool(np.all((amt >= lo) & (amt <= hi)) and
                np.all((prob >= plo) & (prob <= phi)) and
                np.all(amt > 0) and np.all((prob > 0) & (prob < 1)))


def verify_bank(x, amount_range=DEFAULT_AMOUNT_RANGE,
                prob_range=DEFAULT_PROB_RANGE):
    """Legacy support check for a 500-row bank; new CR4 does not use one."""
    return np.asarray(x).shape == (N_TRIALS, 6) and verify_x(
        x, amount_range, prob_range)


def design(args):
    raise RuntimeError("CR4 no longer uses an S0 fixed bank. "
                       "Run --stage dataset for per-subject/per-trial IID X.")


def dataset(args):
    if DATASET.exists():
        raise FileExistsError(
            f"Existing CR4 dataset preserved: {DATASET}. "
            "Archive old outputs/choice/CR4 before regenerating IID-X CR4; "
            "old S2/checkpoints are incompatible.")
    amount_range = (args.amount_min, args.amount_max)
    prob_range = (args.prob_min, args.prob_max)
    x_seed = args.data_seed + 2004
    x = sample_subject_x(args.subjects, seed=x_seed,
                         amount_range=amount_range, prob_range=prob_range)
    if not verify_x(x, amount_range, prob_range):
        raise AssertionError("CR4 IID-X support check failed")
    rng_theta = np.random.default_rng(args.data_seed + 4)
    rng_choice = np.random.default_rng(args.data_seed + 1004)
    theta = sample_theta(args.subjects, rng_theta)
    p = expit(logits(x, theta[:, None, :]))
    choice = (rng_choice.random(p.shape) < p).astype(np.int64)
    if not np.all(np.isfinite(p) & (p >= 0) & (p <= 1)):
        raise AssertionError("CR4 invalid Bernoulli probabilities")
    n = len(theta)
    data = GeneratedDataset(
        parameters=theta,
        trial_counts=np.full(n, N_TRIALS, dtype=np.int64),
        subject_index=np.repeat(np.arange(n, dtype=np.int64), N_TRIALS),
        trial_index=np.tile(np.arange(N_TRIALS, dtype=np.int64), n),
        x=x.reshape(-1, 6).copy(),
        y=choice.reshape(-1, 1),
        predictive_true=p.reshape(-1, 1),
    )
    save_dataset(data, DATASET)
    write_json(DATASET.parent / "s1_generate__manifest.json", dict(
        model="CR4", n_parameters=3, fixed_alpha=1.0, subjects=n,
        trials_per_subject=N_TRIALS, x_dim=6, x_shared=False,
        x_sampling_policy=SAMPLING_POLICY,
        parameter_names=list(PARAMETER_NAMES), data_seed=args.data_seed,
        x_seed=x_seed, amount_range=list(amount_range),
        probability_range=list(prob_range),
        dataset_sha256=hash_file(DATASET),
        formula="CR2/CR3 logits with alpha=1 exactly",
        lambda_gamma_tau_priors="unchanged CR2/CR3 marginals",
        decoder_x_preprocessing="theoretical uniform X moments",
    ))
    print(f"Generated CR4 IID X: {n} subjects x {N_TRIALS} trials:", DATASET, flush=True)


def selftest():
    x = sample_subject_x(3, seed=17)
    assert x.shape == (3, N_TRIALS, 6) and verify_x(x)
    assert np.array_equal(x, sample_subject_x(3, seed=17))
    assert not np.array_equal(x[0], x[1])
    assert not np.array_equal(x[0, 0], x[0, 1])
    theta = sample_theta(3, np.random.default_rng(19))
    expected = cr2_logits(x, theta_as_cr3(theta)[:, None, :])
    assert np.allclose(logits(x, theta[:, None, :]), expected)
    print("CR4 IID X and alpha=1 CPT selftest PASS")


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--stage", choices=("design", "dataset", "selftest"), default="dataset")
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
