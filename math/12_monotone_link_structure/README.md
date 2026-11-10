# 12 · Monotone-link-invariant structural screening (WY-only)

This is a **new, independent one-pass method screen** inspired by unknown-link / varying-coefficient / functional-separability literature. It is NOT a modification of the unsuccessful `11` Hessian-sparsity objective and does NOT claim to implement Breiman–Friedman ACE, SKIM-FA, or symbolic regression.

We only have fixed samples `W=[Z,X],Y`; we do **not** know the generating expression or the output link. The *blind* branch loads an existing predictor fitted to WY and works on its predictions. The separate *oracle* branch calls the analytical generator solely to establish whether the proposed mathematical objective even prefers the desired axes. We must run the oracle Gate 2 **first**, and only run the blind comparison if the oracle test passes. A fail is a fail; do not tune it away.

## What is actually implemented?

### Method D: normalized stimulus-effect **projector** (continuous X only)

If `m(W)=h(g(W))` with a **shared**, continuously differentiable, strictly increasing outside transformation `h`, then wherever `grad_X m` is nonzero,

```
q(W) = grad_X m(W) / ||grad_X m(W)||
P(W) = q(W) q(W)^T
```

is **exactly independent of h**. `P` is additionally unchanged by the *sign* of a common effect amplitude, making it more robust than `q` for Toy A/B. Estimate `P` using central finite differences of a WY-trained smooth predictor. Rotate **only Z**, transform the queried coordinates back to the original coordinates of the **fixed** predictor, and score each candidate axis:

```
a_i(R) = sqrt(E[ || d P_R(U,X) / d U_i ||_F^2 ])
p_i(R) = a_i(R)^2 / sum_j a_j(R)^2
C_eff(R) = exp(-sum_i p_i(R) log(p_i(R)))
C_l1(R) = sum_i a_i(R)
```

A low `C_eff` means the **relative stimulus-effect direction** depends on fewer latent directions; it does *not* mean the whole function depends on fewer dimensions. In Toy A, the common factor `z1` cancels; in Toy B, the relative effect depends on `(z1-z2)` and `z3`. This is precisely the restricted kind of structure the method can detect. The operation is conditional-mean geometry, not a claim to identify the unique internal computation graph.

**Limitations:** needs ≥2 *continuous* X dimensions, a smooth sufficiently well-estimated conditional mean, nonzero X gradients, and a *shared* monotone outer link. It does NOT cancel different nonlinear links nested inside different branches. Finite differences of a fitted predictor can be biased despite excellent held-out R²; oracle success is not enough. Toy X perturbations can slightly leave support near the edges. Never apply to discrete CR3 lottery X coordinates.

### Method O: **ordinal valid-stimulus contrasts** (including discrete CR3 X)

For genuinely observed stimulus vectors `X_a, X_b`,

```
s(Z, Xa, Xb) = sign(m(Z,Xa)-m(Z,Xb))
```

is invariant to an arbitrary **shared strictly increasing transformation** of the response. For each latent axis, measure the fraction of comparisons whose sign differs between `Z+delta*e_i` and `Z-delta*e_i`. Use entropic effective dimension and L1 of these flip fractions. CR3 comparisons only use observed, valid lottery stimuli and **never** perturb X off its support.

**Limitations:** ranking throws away information about strength and common amplitude. Zero flip rate means `NO_SIGNAL`, not success. Oracle Toy B might fail to favor 45°; **if so, reject it immediately**. This is a pairwise ranking sensitivity heuristic, not ACE, not a full conditional-independence test, and not a universal causal interaction estimator.

## Strict one-pass gates

1. **Gate 2 (Oracle mathematical criterion):** does the candidate complexity metric prefer Toy A `0°` or Toy B `45°` (modulo signed permutations, period 90°)? Is the oracle improvement over a contrasting control angle ≥3%? If NO, reject this metric—**do not retrain**.
2. **Gate 3 (WY estimation):** hold the predictor fixed, rotate only Z coordinates through an exact inverse mapping, compare WY-estimated and oracle *curves* on identical held-out points. Require Spearman ≥0.8 and optimal angle error ≤10°. If NO, reject this predictor+metric combination. **No architecture search or hyperparameter chasing in this screening stage.**
3. Only methods that pass both Toy A/B go to C1/C2/C3 and CR3. For CR3, an arbitrary 45° is not a known correct target, so evaluate agreement with the oracle *objective*, not identity-axis recovery.

`compare.py` records `PASS / FAIL / N/A` without changing any settings. This is a **screening threshold**, not a statistical significance test. Note that empirical complex structured functions need not have a unique sparse coordinate system.

## CMD: first screen Toy B with your existing MLP (no retraining)

Run these commands in the root of `mod-rot-3` (Windows CMD). The earlier `outputs/wybench/toy_b` and `outputs/wybench/toy_b/mlp/model.pt` must already exist; they are NOT in Git.

```bat
conda activate py11
git pull
python -m pytest -q math/12_monotone_link_structure/tests
```

**Step B1 — Oracle only (Gate 2)**

```bat
python math/12_monotone_link_structure/run.py --data outputs/wybench/toy_b --method direction --oracle-scale prob --axis-i 0 --axis-j 1 --step 5 --n-eval 128 --output outputs/wybench/toy_b/link12_oracle
```

Inspect `outputs/wybench/toy_b/link12_oracle/summary.json`. An optimal angle near `45` or `-45` (or their 90° equivalents) supports moving to B2. If not, **stop and upload only the Oracle folder**.

**Step B2 — WY-only score (same frozen original MLP; no further training)**

```bat
python math/12_monotone_link_structure/run.py --data outputs/wybench/toy_b --fit outputs/wybench/toy_b/mlp --method direction --axis-i 0 --axis-j 1 --step 5 --n-eval 128 --output outputs/wybench/toy_b/link12_wy
python math/12_monotone_link_structure/compare.py --estimated outputs/wybench/toy_b/link12_wy/rotation_curve.json --oracle outputs/wybench/toy_b/link12_oracle/rotation_curve.json --output outputs/wybench/toy_b/link12_compare
```

**UPLOAD**: `toy_b_link12_screen.zip`, created using:

```bat
tar -a -cf toy_b_link12_screen.zip -C outputs/wybench/toy_b link12_oracle link12_wy link12_compare mlp/fit.json
```

The ZIP contains all curves, per-axis strengths, gate report, and training metadata, **not** the heavy MLP checkpoint or raw data. If stopping after Gate 2, use:

```bat
tar -a -cf toy_b_link12_oracle_only.zip -C outputs/wybench/toy_b link12_oracle
```

### Then Toy A (only if Toy B succeeded)

```bat
python math/11_wy_nonlinear_interactions/data.py --model toy_a --label prob --subjects 180 --trials 40 --output outputs/wybench/toy_a_link12
python math/11_wy_nonlinear_interactions/models.py --data outputs/wybench/toy_a_link12 --method mlp --epochs 200 --output outputs/wybench/toy_a_link12/mlp
python math/12_monotone_link_structure/run.py --data outputs/wybench/toy_a_link12 --method direction --oracle-scale prob --step 5 --n-eval 128 --output outputs/wybench/toy_a_link12/link12_oracle
python math/12_monotone_link_structure/run.py --data outputs/wybench/toy_a_link12 --method direction --fit outputs/wybench/toy_a_link12/mlp --step 5 --n-eval 128 --output outputs/wybench/toy_a_link12/link12_wy
python math/12_monotone_link_structure/compare.py --estimated outputs/wybench/toy_a_link12/link12_wy/rotation_curve.json --oracle outputs/wybench/toy_a_link12/link12_oracle/rotation_curve.json --output outputs/wybench/toy_a_link12/link12_compare
```

**UPLOAD** Toy A result folder ZIP (after run):

```bat
tar -a -cf toy_a_link12_screen.zip -C outputs/wybench/toy_a_link12 link12_oracle link12_wy link12_compare mlp/fit.json
```

### Separate ordinal trial, only to decide if this metric survives Gate 2

```bat
python math/12_monotone_link_structure/run.py --data outputs/wybench/toy_b --method ordinal --oracle-scale prob --axis-i 0 --axis-j 1 --step 10 --n-eval 64 --pairs 48 --output outputs/wybench/toy_b/link12_ordinal_oracle
```

Upload `outputs/wybench/toy_b/link12_ordinal_oracle/` only if you want the ordinal candidate audited. **Do not bother training a new model for it before Oracle supports the intended orientation.**

### CR3 (only for methods surviving Toy A/B)

All real `X` vectors are taken from your existing valid fixed lottery bank, even in model evaluations; no arbitrary X perturbation. Use `--method ordinal --standardize-z` because the four CR3 parameter prior scales differ. Example true-coordinate oracle criterion (not yet a claim of recovery):

```bat
python math/11_wy_nonlinear_interactions/data.py --model CR3 --label prob --subjects 500 --trials 200 --output outputs/wybench/CR3_link12
python math/12_monotone_link_structure/run.py --data outputs/wybench/CR3_link12 --method ordinal --oracle-scale prob --axis-i 1 --axis-j 3 --step 10 --n-eval 64 --pairs 48 --standardize-z --output outputs/wybench/CR3_link12/oracle_lambda_tau
```

Upload `outputs/wybench/CR3_link12/oracle_lambda_tau/` as `cr3_link12_oracle.zip` if you run it:

```bat
tar -a -cf cr3_link12_oracle.zip -C outputs/wybench/CR3_link12 oracle_lambda_tau
```

For binary choice data, `m(W)=P(Y=1|W)` must be learned from WY before applying these scores. A single observed 0/1 Y cannot be numerically transformed into a true probability or logit. `--label prob` is a **simulation diagnostic** where exact probabilities are supplied as labels; `--label bernoulli` is the realistically noisy case.

## Outputs / audit

- `rotation_curve.json`: full angle grid, score strengths for **each Z axis**, score validity, sample sizes, quality checks, method and output scale.
- `rotation_curve.csv`: spreadsheet-friendly scores.
- `effective_axes.png`, `l1.png`: the two structural losses.
- `summary.json`: minima and finite-score counts.
- `gate_report.json`, `comparison.png`: score-by-score PASS / FAIL / N/A and direct estimated-versus-oracle curve comparison.

The blind branch (`--fit`) imports only saved WY-trained regression functions, never `oracle.py` or the generators. The oracle branch (`--oracle-scale`) is visibly separate. All code lives only under `math/12_monotone_link_structure/`; original pipelines and prior experiments are untouched.

## Literature context / limitations

This experiment is informed by unknown-link regression and functional separability, but is not a published standalone estimator. The mathematical outer-link invariance follows by the chain rule (projector) and preservation of rankings under strictly increasing functions (ordinal). References to investigate independently: Breiman & Friedman (1985) ACE; Horowitz & Mammen (2007) nonparametric unknown-link regression; Fan & Zhang (2008) varying-coefficient models; Udrescu & Tegmark (2020) AI Feynman; Ba et al. (2024) sparse additive decomposition under basis transforms. This experiment intentionally tests **one narrowly defined structure hypothesis** rather than claiming to solve general symbolic recovery.
