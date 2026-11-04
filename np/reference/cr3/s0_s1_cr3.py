"""CR3 4P: same CR2 probability model/prior, different fixed 500-trial bank.

No NP is needed to design or audit the stimulus bank. In particular, no
new Fisher/choice optimization is performed. General trials are sampled
without replacement from the existing optimized CR2-4P bank.

First: --stage design. Optional, LATER: --stage dataset (5k subjects).
CR2 assets are only read; never modified.
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
ROOT = HERE.parents[1]
sys.path[:0] = [str(RESEARCH), str(ROOT / "src")]
from s0_cr2_design_recovery import sample_prior, logits, NAMES, make_candidates
from modlearn_rot.generators.dataset import GeneratedDataset, save_dataset

OUT = ROOT / "outputs" / "choice" / "CR3"
BANK = OUT / "s0_design" / "CR3_4P_fixed500.npz"
DATASET = OUT / "s1_generate" / "s1_generate__dataset.npz"
ORIGINAL = (ROOT / "outputs" / "choice" / "CR2" / "s0_design_recovery"
            / "CR2_4P_seed20261023_fixed500.npz")

COUNTS = {"gain_only": 60, "gamma_neutral": 60,
          "loss_only": 40, "general_CR2": 340}
CODES = {name: j for j, name in enumerate(COUNTS)}
AMOUNTS = np.array([5, 10, 15, 20, 30, 40, 50, 60, 80, 100, 125, 150, 175, 200.])
PROBABILITIES = np.array([.05, .1, .15, .2, .25, .3, .35, .4, .45,
                          .5, .55, .6, .65, .7, .75, .8, .85, .9, .95])


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False),
                    encoding="utf-8")


def condition_counts(x):
    """Disjoint categories; gamma-neutral is exactly both probabilities=.5."""
    gain = np.isclose(x[:, 1], 0) & np.isclose(x[:, 4], 0)
    loss = np.isclose(x[:, 0], 0) & np.isclose(x[:, 3], 0)
    neutral = np.isclose(x[:, 2], .5) & np.isclose(x[:, 5], .5)
    return dict(gain_only=int(gain.sum()), loss_only=int(loss.sum()),
                gamma_neutral=int(neutral.sum()),
                general=int((~(gain | loss | neutral)).sum()))


def canonical(row):
    a = tuple(np.round(row[:3], 10))
    b = tuple(np.round(row[3:], 10))
    return min((a, b), (b, a))


def first_order_dominates(A, B, tol=1e-12):
    """Exact first-order stochastic dominance for two-outcome lotteries.

    Each lottery is (nonnegative gain, nonnegative loss magnitude, p_gain).
    Its outcomes are -loss (probability 1-p_gain) and +gain (p_gain).
    A strictly FOSD B iff CDF_A(t)<=CDF_B(t) for EVERY payoff t, with
    strict inequality for at least one interval. The finite union of
    payoff support points is sufficient for this discrete comparison.

    Unlike a heuristic comparison of amount/probability columns, this
    also handles zero outcomes, sure events, and mixed gains/losses.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != (3,) or B.shape != (3,):
        raise ValueError("FOSD expects two lotteries (gain,loss,p)")
    if (np.any(A[:2] < 0) or np.any(B[:2] < 0) or
        not (0 <= A[2] <= 1 and 0 <= B[2] <= 1)):
        raise ValueError("Invalid lottery outcomes or probabilities")
    support = np.unique([-A[1], A[0], -B[1], B[0]])
    cdf_a = (1-A[2])*(-A[1] <= support) + A[2]*(A[0] <= support)
    cdf_b = (1-B[2])*(-B[1] <= support) + B[2]*(B[0] <= support)
    return bool(np.all(cdf_a <= cdf_b + tol) and np.any(cdf_a < cdf_b - tol))


def dominance_direction(row):
    """1 if option A strictly FOSD B, -1 if B FOSD A, otherwise 0."""
    row = np.asarray(row, dtype=float)
    if row.shape != (6,):
        raise ValueError("Trial must be six-dimensional")
    if first_order_dominates(row[:3],row[3:]):
        return 1
    if first_order_dominates(row[3:],row[:3]):
        return -1
    return 0


def dominance_audit(X):
    """Counts FOSD trials; the CR3 release gate requires zero."""
    X=np.asarray(X, dtype=float)
    directions=np.array([dominance_direction(x) for x in X],dtype=int)
    return dict(total=int(np.count_nonzero(directions)),
                option_a_dominates=int(np.count_nonzero(directions==1)),
                option_b_dominates=int(np.count_nonzero(directions==-1)),
                total_trials=int(len(X)))


def no_strict_fosd(row):
    return dominance_direction(row)==0


def sample_special(kind, rng):
    """No model fitting or Fisher optimization; just randomized valid lotteries."""
    if kind == "gain_only":
        ga, gb = rng.choice(AMOUNTS, 2, replace=False)
        pa, pb = rng.choice(PROBABILITIES, 2, replace=False)
        x = np.array([ga, 0., pa, gb, 0., pb], dtype=np.float64)
        if pa == .5 and pb == .5:
            raise AssertionError("Different probabilities cannot both equal .5")
    elif kind == "gamma_neutral":
        ga, la, gb, lb = rng.choice(AMOUNTS, 4, replace=False)
        x = np.array([ga, la, .5, gb, lb, .5], dtype=np.float64)
    elif kind == "loss_only":
        la, lb = rng.choice(AMOUNTS, 2, replace=False)
        pa, pb = rng.choice(PROBABILITIES, 2, replace=False)
        x = np.array([0., la, pa, 0., lb, pb], dtype=np.float64)
    else:
        raise ValueError(kind)
    if rng.random() < .5:
        x = np.concatenate((x[3:], x[:3]))
    return x


def make_bank(original_x, seed=20261023):
    """Fixed 60+60+40+340 allocation; zero incremental optimization."""
    original_x = np.asarray(original_x, dtype=np.float64)
    if original_x.shape != (500, 6) or not np.isfinite(original_x).all():
        raise ValueError("CR2 fixed bank must be finite (500,6)")
    rng = np.random.default_rng(seed + 3003)
    gain = (original_x[:, 1] == 0) & (original_x[:, 4] == 0)
    loss = (original_x[:, 0] == 0) & (original_x[:, 3] == 0)
    neutral = (original_x[:, 2] == .5) & (original_x[:, 5] == .5)
    pool = original_x[~(gain | loss | neutral)]
    # CR2's original selection did NOT ban strictly dominant trials.
    # Exclude them rather than inheriting obvious choices in CR3.
    clean_pool = np.array([row for row in pool if no_strict_fosd(row)],
                          dtype=np.float64).reshape(-1, 6)
    order = rng.permutation(len(clean_pool))
    chosen = clean_pool[order[:COUNTS["general_CR2"]]]
    rows = [r.copy() for r in chosen]
    labels = [CODES["general_CR2"]] * len(chosen)
    seen = {canonical(row) for row in rows}
    if len(seen) != len(rows):
        raise ValueError("Original CR2 selection contains mirrored duplicates")
    # If too many old CR2 items are FOSD, replenish with unbiased draws
    # from the SAME original CR2 lottery candidate generator; no fitting.
    extra_general = 0
    rounds = 0
    while len(rows) < COUNTS["general_CR2"]:
        rounds += 1
        if rounds > 100:
            raise RuntimeError("Could not generate enough general no-FOSD stimuli")
        candidates, _ = make_candidates(500, rng)
        for row in candidates:
            if len(rows) == COUNTS["general_CR2"]:
                break
            ga = (row[1] == 0 and row[4] == 0)
            lo = (row[0] == 0 and row[3] == 0)
            gn = (row[2] == .5 and row[5] == .5)
            key = canonical(row)
            if ga or lo or gn or key in seen or not no_strict_fosd(row):
                continue
            seen.add(key)
            rows.append(row)
            labels.append(CODES["general_CR2"])
            extra_general += 1
    for kind in ("gain_only", "gamma_neutral", "loss_only"):
        accepted = 0
        attempts = 0
        while accepted < COUNTS[kind]:
            attempts += 1
            if attempts > 100000:
                raise RuntimeError("Unique non-dominant CR3 special-condition sampling stalled")
            row = sample_special(kind, rng)
            key = canonical(row)
            if key in seen or not no_strict_fosd(row):
                continue
            seen.add(key)
            rows.append(row)
            labels.append(CODES[kind])
            accepted += 1
    x = np.stack(rows)
    labels = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(len(x))
    x, labels = x[order], labels[order]
    checks = condition_counts(x)
    for key in ("gain_only", "loss_only", "gamma_neutral"):
        if checks[key] != COUNTS[key]:
            raise AssertionError(f"CR3 {key} count mismatch: {checks}")
    if checks["general"] != COUNTS["general_CR2"] or len(x) != 500:
        raise AssertionError("CR3 500-trial composition mismatch")
    if len({canonical(r) for r in x}) != 500:
        raise AssertionError("Duplicate or mirrored CR3 stimuli")
    if dominance_audit(x)["total"] != 0:
        raise AssertionError("Strictly dominant lottery pair found in CR3")
    return x, labels


def design(args):
    if BANK.exists() and not args.overwrite:
        raise FileExistsError(f"CR3 bank already exists: {BANK}; use --overwrite to replace")
    if DATASET.exists():
        raise FileExistsError(
            "CR3 subject dataset already exists and may use the OLD stimulus bank: "
            f"{DATASET}. Archive the entire outputs/choice/CR3 directory before "
            "changing its fixed bank, then regenerate the design/dataset and "
            "retrain any CR3 NP checkpoint. Do not reuse old CR3 checkpoints.")
    source = Path(args.original_bank)
    if not source.is_file():
        raise FileNotFoundError(f"Missing CR2 fixed bank: {source}. "
                                "Run CR2 s0 design or supply --original-bank")
    with np.load(source, allow_pickle=False) as f:
        x = np.asarray(f["X"], dtype=np.float64)
    source_dominance=dominance_audit(x)
    bank, labels = make_bank(x, args.seed)
    BANK.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(BANK, X=bank, type_codes=labels,
                        type_names=np.asarray(list(COUNTS)),
                        parameter_names=np.asarray(NAMES[4]))
    rng = np.random.default_rng(args.seed + 111)
    test_theta = sample_prior(8, 4, rng)
    pr = expit(logits(bank[None], test_theta[:, None]))
    if not np.isfinite(pr).all() or not ((pr > 0) & (pr < 1)).all():
        raise AssertionError("CR3 probability selfcheck failed")
    write_json(BANK.parent / "CR3_4P_design.json", dict(
        model="CR3_4P", parent_model="CR2_4P",
        formula="logit P(A)=tau*(V(A)-V(B)), exactly CR2-4P",
        parameters=list(NAMES[4]), seed=args.seed, trial_count=500,
        counts=COUNTS, measured_counts=condition_counts(bank),
        strict_first_order_dominance_forbidden=True,
        baseline_dominance=source_dominance,
        cr3_dominance=dominance_audit(bank),
        baseline_cr2_bank=str(source), baseline_sha256=hash_file(source),
        new_bank_sha256=hash_file(BANK),
        design="No new objective or Fisher optimization. Up to 340 selected at random from nondominant original CR2 general trials; if needed, additional general lottery candidates sampled to refill. 160 randomly sampled nondominant special conditions.",
        subject_prior="sample_prior(4) imported unchanged from s0_cr2_design_recovery",
        special_proportions=dict(gain=.12, gamma_neutral=.12, loss=.08),
        warning="This bank is CR3-only. It does NOT overwrite or modify CR2."))
    print(f"CR3 fixed 500-trial bank: {BANK}", flush=True)
    print("Stimulus coverage:", condition_counts(bank), flush=True)
    print("FOSD audit: CR2 =", source_dominance["total"],
          "; CR3 =", dominance_audit(bank)["total"], flush=True)


def dataset(args):
    if DATASET.exists() and not args.overwrite:
        raise FileExistsError(f"{DATASET} already exists; --overwrite required")
    if not BANK.exists():
        raise FileNotFoundError(f"First create CR3 bank: {BANK}")
    with np.load(BANK, allow_pickle=False) as f:
        x = np.asarray(f["X"], float)
    if x.shape != (500, 6):
        raise ValueError("CR3 bank must be 500x6")
    if dominance_audit(x)["total"] != 0:
        raise ValueError("CR3 bank has strictly dominated trials. "
                         "Regenerate its fixed bank with updated s0_s1_cr3.py")
    # Use exact CR2 parameter/choice streams (same seeds for paired comparison).
    rng_theta = np.random.default_rng(args.data_seed + 4)
    rng_choice = np.random.default_rng(args.data_seed + 1004)
    theta = sample_prior(args.subjects, 4, rng_theta)
    p = expit(logits(x[None, :, :], theta[:, None, :]))
    choice = (rng_choice.random(p.shape) < p).astype(np.int64)
    n = len(theta)
    ds = GeneratedDataset(
        parameters=theta,
        trial_counts=np.full(n, 500, dtype=np.int64),
        subject_index=np.repeat(np.arange(n, dtype=np.int64), 500),
        trial_index=np.tile(np.arange(500, dtype=np.int64), n),
        x=np.broadcast_to(x, (n, 500, 6)).reshape(-1, 6).copy(),
        y=choice.reshape(-1, 1),
        predictive_true=p.reshape(-1, 1),
    )
    save_dataset(ds, DATASET)
    write_json(DATASET.parent / "s1_generate__manifest.json", dict(
        model="CR3_4P", subjects=n, trials_per_subject=500,
        parameter_names=list(NAMES[4]), data_seed=args.data_seed,
        bank_sha256=hash_file(BANK), dataset_sha256=hash_file(DATASET),
        cr2_formula_and_priors_unchanged=True, neural_network_trained=False))
    print(f"CR3 dataset generated (no NP training): {DATASET}", flush=True)


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--stage", choices=("design", "dataset", "all", "selftest"),
                   default="design")
    p.add_argument("--original-bank", default=str(ORIGINAL))
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--data-seed", type=int, default=20261028)
    p.add_argument("--subjects", type=int, default=5000)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.subjects < 1:
        p.error("subjects must be positive")
    if args.stage == "selftest":
        # Both explicit example types and their inverse trade-offs.
        cases = [
            ([100,0,.75, 80,0,.5], True),
            ([0,20,.8, 0,30,.6], True),
            ([100,20,.5, 80,30,.5], True),
            ([100,0,.3, 80,0,.8], False),
            ([0,30,.9, 0,20,.5], False),
            ([100,60,.6, 80,20,.5], False),
        ]
        for row, expected in cases:
            if (dominance_direction(row) != 0) != expected:
                raise AssertionError(f"Dominance selftest failed: {row}")
            if (dominance_direction(row[3:]+row[:3]) != 0) != expected:
                raise AssertionError(f"Reverse dominance selftest failed: {row}")
        with np.load(args.original_bank, allow_pickle=False) as f:
            baseline = f["X"]
        x, labels = make_bank(baseline, args.seed)
        assert x.shape == (500, 6) and len(labels) == 500
        assert np.array_equal(x, make_bank(baseline, args.seed)[0])
        assert dominance_audit(x)["total"] == 0
        print("CR3 deterministic 500-X bank, FOSD and composition selftest PASS")
        return
    if args.stage in ("design", "all"):
        design(args)
    if args.stage in ("dataset", "all"):
        dataset(args)


if __name__ == "__main__":
    main()
