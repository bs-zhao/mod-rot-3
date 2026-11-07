# 09 — CR3 three-gap audit of the EXISTING successful math/07 indicators

This experiment is deliberately **not a new rotation metric**. It reads prior math/07 output and reuses its three sources: (1) exact analytic CR3; (2) the same indicator computed from regression on numerical formula-generated WY; (3) exactly the same regression/indicator computed from frozen decoder-generated WY. It does not call the generator, retrain the NP, refit any surrogate, or import other math experiment code.

## Motivation: preserve what math/07 ALREADY found

The original math/07 result includes 40 candidates meeting its 6/6 analytic near-endpoint criterion (21 logit and 19 probability). Several survived the formula-Y evaluation, but none met all six decoder-Y comparisons. Never state that no analytic indicators work simply because the narrower math/08 Z–X Soft-L0 test fails.

A critical clarification: in math/07 **success means within 10 degrees of 0 or 90**, NOT a strict 0/90 minimum. Original screening also used relative curve range >= 0.02 and Pearson >= 0.9. This code reproduces that distinction and adds a STRICT-endpoint audit in parallel.

Some established math/07 examples:

- Logit mixed-Z third derivative mean absolute loss: six-plane oracle 6/6, formula-Y 6/6, decoder-Y 3/6.
- Logit between-subject variance of diagonal ZZ curvature: 6/6, 6/6, 3/6.
- Probability ZZ off-diagonal magnitude Q99 loss: 6/6, 6/6, 4/6.
- Probability ZZ RMS flat next-k1 loss: 6/6, 6/6, 1/6.

The full audit is **exclusively read-only** on existing 07 metrics. It maintains the *same mathematical metric* for all three gaps. It runs no parameter-axis optimizer on decoder Y here: the original experiment used fixed rotations of its estimated A-star map and compared loss landscapes. It must not be represented as a global SO(4) parameter recovery experiment.

## Input format

Provide the complete output folder of math/07 (the directory holding strict_screen_all.csv, summary.json and all_local_metrics.csv), or a ZIP containing that directory.

The output of 07 normally lives under:

outputs/math/07_cr3_local_hessian_complexity/<condition>/repeat01/<estimator>/

The source archive may contain a single enclosing directory and is supported directly; no ZIP extraction is required.

## Windows CMD

Activate the existing Python environment and pull the new code:

    conda activate py11
    cd /d D:\Projects\mod-rot-3
    git pull --ff-only
    python -m unittest discover -s math/09_cr3_three_gap_reaudit -p "test_*.py" -v

Full audit of the previously saved 07 output ZIP (replace the filename with its actual location):

    python math/09_cr3_three_gap_reaudit/run.py --source "D:\path\to\monomial_u4_x4_lambu1e-06_lambx1e-06.zip"

Or use the existing 07 output directory directly:

    python math/09_cr3_three_gap_reaudit/run.py --source "outputs\math\07_cr3_local_hessian_complexity\<condition>\repeat01\<estimator>"

Faster text-only mode, skipping plots:

    python math/09_cr3_three_gap_reaudit/run.py --source "D:\path\to\07.zip" --output outputs/math/09_quick --top-plots 0

## Outputs

- all_plane_gap_checks.csv: each metric × target × plane; analytic, formula and decoder minima, curve correlations and sequential gate results.
- all_metrics_ranked.csv: all old metrics ranked by near-endpoint Gap 3, then Gap 2 and Gap 1, also strict scores.
- gap1_near_six_of_six.csv, gap2_near_six_of_six.csv, gap3_near_six_of_six.csv.
- gap1_strict_six_of_six.csv, gap2_strict_six_of_six.csv, gap3_strict_six_of_six.csv.
- report.md: summary of candidates surviving each stage.
- manifest.json: A-star from source 07 and the explicit fixed thresholds.
- plots/: six-plane three-source landscapes for top near-oracle candidates.

## Interpretation and limitations

1. Gap 1: analytic indicator produces a nonflat curve with endpoint or near-endpoint minimum.
2. Gap 2: same indicator computed ONLY from numeric formula WY also produces an endpoint minimum and correlates >= threshold with analytic curve.
3. Gap 3: same estimator/indicator computed with decoder-Y does so as well, conditional on the earlier gates.
4. Success in six fixed two-dimensional Givens planes does **NOT** prove a minimum throughout full SO(4). That should be a *subsequent* test for the most promising historically supported indicators.
5. The old 07 source is anchored via T-hat=A-star Z+b; no true T enters rotation training, but the reference A-star is a supervised oracle diagnostic.
6. Analysis on the same outputs used to discover the metrics is not independent confirmation. A future research claim requires a held-out simulation and consistent settings across C1/C2/C3/CR3.
7. Choice (0/1) is deliberately excluded, since its non-smooth and stochastic target differs from the logit/probability functions. It needs a separate choice-observation protocol.

## What comes next

Use this audit to prioritize existing Gap-1/Gap-2 survivors, not to invent an unrelated indicator. The next experiments should target robust decoder-Y recovery and full SO(4) validation on fresh input draws, keeping mathematical target, normalization, data support and estimation protocol fixed.
