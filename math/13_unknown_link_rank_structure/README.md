# 13 · Unknown-link rank interaction structure — C1/C2/C3 **and CR3**

**CORRECTNESS UPDATE (2026-11-10):** `scan.py` v1 accidentally omitted the upper off-diagonal `sin(angle)` term from its Givens rotation. The old matrices were not orthogonal, particularly at ±90°, and caused spurious structural minima. **All rotation curves/gates generated before the corrected v2 commit are invalid.** Models and polynomial fits do NOT need retraining: rerun `scan.py`, `gate2.py`, and `compare.py` only. The latter scripts now reject legacy scan JSON without `rotation_matrix_type=orthogonal_givens_v2`. CR3 original probability-condition results after re-evaluation: Oracle R²=.9827; Oracle gain normalized_l1=1.36%, effective_partner_sum=1.91%; both fail preset 3% Gate 2 threshold; Gate 3 fails as well. No hyperparameters were changed.


**Research question:** From paired samples `W=[Z,X], Y`, can a **single-index model with an unknown common strictly increasing output link** recover a simple, coordinate-dependent `Z × X` interaction structure? Does the selected rotation agree with a separately calculated oracle? This is a **new, standalone, one-pass method screen**, not an attempt to repair the failed Hessian or direction-projector objectives.

## What is implemented (and what is NOT)

**Blind WY-only model:** Let `u=standardize(Z)` and let `ψ_z(u)` and `φ_x(standardize(X))` be *prespecified data-derived monomial bases*. Fit a restricted index

```
g_hat(W) = a + A · ψ_z(u) + b · φ_x(X) + ψ_z(u)^T C φ_x(X)
Y ≈ h(g_hat(W))
```

with an **unknown strictly increasing** outer `h`. `fit.py --method rank_svm` learns the direction of the index coefficients from pairwise Y orderings using a linear SVM on feature differences, an inexpensive **rank-surrogate** for semiparametric single-index estimation. Then isotonic regression estimates the output calibration on the train split; report held-out rank accuracy AND calibrated prediction R². For every shared strictly increasing `h`, pairwise orderings are unchanged. This method requires an approximately representable *single latent index* and a continuous, reasonably informative response for training ranks. `--method ridge_y` is a link-naive baseline. **Neither method is the original ACE, GSAMUL, NestyNet, or the exact Han maximum rank correlation estimator.**

**Objective:** For the learned polynomial index, compute `G[n,i,k]=∂g_hat(Z_n,X)/∂u_i`, resolved into stimulus dictionary coefficients `k` at the **same held-out Z anchors**. Rotate `u` by a Givens matrix `R` and use `R G`; for each `(i,k)` take RMS across anchors. The reported L1 is the sum of these RMS values **normalized by the global Frobenius energy** (rotation invariant). Also report the effective count of latent partners per stimulus feature. `scan.py` never regenerates Y, never opens `oracle_reference.npz`, never differentiates X, and never retrains for a new angle. These are *functional regression component loadings*, not universal reconstruction of an unknown computation graph. The stimulus dictionary is fixed, so coordinate-wise interaction counts depend on its choice.

**CR3 is first-class:** CR3 true CPT parameters are 4D, stimuli have **6 discrete lottery attributes**. It runs the **same** rank-based candidate, with degree 2 Z and degree 2 X polynomial features by default; no illegal partial derivative in lottery X, no synthetic feature changes of the actual lotteries. The X dictionary is formed from real observed valid bank vectors. `oracle.py` separately fits a polynomial projection to the true **internal logit** and evaluates held-out approximation R². This projection is **approximate** for CR3, so if Oracle R² < **0.90**, **FAIL Gate 2**: do NOT inflate the basis or retune as part of this screen. CR3 has nonisotropic theta priors; no universal theorem that raw coordinates minimize this specific complexity. CR3's Gate 2 tests whether the Oracle structural landscape is informative, not whether the optimum is exactly zero degrees.

**Two Y conditions, critically different:**

- `--label prob`: **simulation-only** deterministic true conditional choice probabilities supplied as Y, so the unknown-link rank estimator can use observed continuous Y; even though the generator created Y, the method does **not** access the generating equation. This is not the real 0/1 choice-observation setting.
- `--label bernoulli`: real binary choice labels. Binary 0/1 labels **cannot** supply an ordering of the conditional mean, so first train a standard `mlp` classifier on training subjects from `(W,choice)`, using the existing 11 script. Pass the trained checkpoint folder via `fit.py --source-fit`. The rank estimator only sees that WY-trained surrogate probability, never true per-trial probabilities; it reports this distinct condition. Never treat all-trial NP Z as clean for same-trial target evaluation (use the context-only extraction from 11).

## Strict gates, without post-failure tuning

Gate 2 checks the **ORACLE**: fit the restricted polynomial-index dictionary to the true internal logit and require held-out R² ≥ **0.90**, an informative rotation gain ≥ **3%**, and for C1/C2/C3 the optimum must be within **10°** of the original parameter axes. For CR3, do not invent a 'true best angle': only demand non-flatness and adequate projection fit. Gate 3 requires WY/Oracle angle error ≤ **10°** modulo signed permutation, rank-curve Spearman ≥ **0.80**, and WY held-out rank accuracy ≥ **0.75** (relative to the available continuous Y or, in the Bernoulli condition, the independently trained surrogate). These are fixed *screening* thresholds, not statistical guarantees.

For any model, if Gate 2 fails, stop; **do not train a WY model to repair the candidate**. If Gate 3 fails, report failure and move on to another method. Do not fit the link using oracle logits, do not select a polynomial degree based on Oracle success, and do not alter the thresholds after seeing results.

## Windows CMD — first tests: C1 and CR3

Run from the repository root. Scripts are independent; **no run-all launcher**. `math/11` must already exist (this package reuses its dataset prep and old generator in the separated oracle stage).

```bat
conda activate py11
git pull
python -m pytest -q math/13_unknown_link_rank_structure/tests
```

### C1 Oracle first (do not train if oracle fails)

```bat
if not exist "outputs\choice\C1\s1_generate\s1_generate__dataset.npz" python np/s1_generate.py --model C1
python math/11_wy_nonlinear_interactions/data.py --model C1 --label prob --subjects 200 --trials 60 --output outputs/wy13/C1_prob
python math/13_unknown_link_rank_structure/oracle.py --data outputs/wy13/C1_prob --z-degree 1 --x-degree 2 --output outputs/wy13/C1_prob/oracle
python math/13_unknown_link_rank_structure/scan.py --fit outputs/wy13/C1_prob/oracle --axis-i 1 --axis-j 2 --step 5 --output outputs/wy13/C1_prob/oracle/scan
python math/13_unknown_link_rank_structure/gate2.py --oracle-fit outputs/wy13/C1_prob/oracle --output outputs/wy13/C1_prob/oracle/scan
```

If Oracle R² ≥ .90 and optimum near 0 with at least 3% gain over 45°, continue:

```bat
python math/13_unknown_link_rank_structure/fit.py --data outputs/wy13/C1_prob --method rank_svm --z-degree 1 --x-degree 2 --output outputs/wy13/C1_prob/wy_rank
python math/13_unknown_link_rank_structure/scan.py --fit outputs/wy13/C1_prob/wy_rank --axis-i 1 --axis-j 2 --step 5 --output outputs/wy13/C1_prob/wy_rank/scan
python math/13_unknown_link_rank_structure/compare.py --estimated outputs/wy13/C1_prob/wy_rank/scan/rotation_curve.json --oracle outputs/wy13/C1_prob/oracle/scan/rotation_curve.json --output outputs/wy13/C1_prob/comparison
```

**UPLOAD if all ran:** `C1_wy13_screen.zip`, made with:

```bat
tar -a -cf C1_wy13_screen.zip -C outputs/wy13/C1_prob oracle wy_rank comparison
```

**UPLOAD if Oracle failed:** `C1_wy13_oracle.zip`, made with:

```bat
tar -a -cf C1_wy13_oracle.zip -C outputs/wy13/C1_prob oracle
```

### CR3 Oracle first — CR3 is an independent required test, even if C1 failed

CR3 has nonisotropic parameter scales. This scan rotates standardized true parameter coordinates; it does *not* claim rotation in the physical units of CPT parameters.

```bat
if not exist "outputs\choice\CR3\s1_generate\s1_generate__dataset.npz" python np/s1_generate.py --model CR3
python math/11_wy_nonlinear_interactions/data.py --model CR3 --label prob --subjects 260 --trials 90 --output outputs/wy13/CR3_prob
python math/13_unknown_link_rank_structure/oracle.py --data outputs/wy13/CR3_prob --z-degree 2 --x-degree 2 --output outputs/wy13/CR3_prob/oracle
python math/13_unknown_link_rank_structure/scan.py --fit outputs/wy13/CR3_prob/oracle --axis-i 0 --axis-j 2 --step 5 --output outputs/wy13/CR3_prob/oracle/scan
python math/13_unknown_link_rank_structure/gate2.py --oracle-fit outputs/wy13/CR3_prob/oracle --output outputs/wy13/CR3_prob/oracle/scan
```

If the CR3 Oracle projection R² ≥ .90 and the gain across angles ≥ 3%, then:

```bat
python math/13_unknown_link_rank_structure/fit.py --data outputs/wy13/CR3_prob --method rank_svm --z-degree 2 --x-degree 2 --output outputs/wy13/CR3_prob/wy_rank
python math/13_unknown_link_rank_structure/scan.py --fit outputs/wy13/CR3_prob/wy_rank --axis-i 0 --axis-j 2 --step 5 --output outputs/wy13/CR3_prob/wy_rank/scan
python math/13_unknown_link_rank_structure/compare.py --estimated outputs/wy13/CR3_prob/wy_rank/scan/rotation_curve.json --oracle outputs/wy13/CR3_prob/oracle/scan/rotation_curve.json --output outputs/wy13/CR3_prob/comparison
```

**UPLOAD if all ran:** `CR3_wy13_prob_screen.zip`, made with:

```bat
tar -a -cf CR3_wy13_prob_screen.zip -C outputs/wy13/CR3_prob oracle wy_rank comparison
```

**UPLOAD if Oracle failed:** `CR3_wy13_oracle.zip`, made with:

```bat
tar -a -cf CR3_wy13_oracle.zip -C outputs/wy13/CR3_prob oracle
```

### CR3 binary-choice WY condition — only if CR3 probability condition passed

```bat
python math/11_wy_nonlinear_interactions/data.py --model CR3 --label bernoulli --subjects 260 --trials 90 --output outputs/wy13/CR3_choice
python math/11_wy_nonlinear_interactions/models.py --data outputs/wy13/CR3_choice --method mlp --epochs 200 --output outputs/wy13/CR3_choice/choice_mlp
python math/13_unknown_link_rank_structure/oracle.py --data outputs/wy13/CR3_choice --z-degree 2 --x-degree 2 --output outputs/wy13/CR3_choice/oracle
python math/13_unknown_link_rank_structure/scan.py --fit outputs/wy13/CR3_choice/oracle --axis-i 0 --axis-j 2 --step 5 --output outputs/wy13/CR3_choice/oracle/scan
python math/13_unknown_link_rank_structure/fit.py --data outputs/wy13/CR3_choice --method rank_svm --source-fit outputs/wy13/CR3_choice/choice_mlp --z-degree 2 --x-degree 2 --output outputs/wy13/CR3_choice/wy_rank
python math/13_unknown_link_rank_structure/scan.py --fit outputs/wy13/CR3_choice/wy_rank --axis-i 0 --axis-j 2 --step 5 --output outputs/wy13/CR3_choice/wy_rank/scan
python math/13_unknown_link_rank_structure/compare.py --estimated outputs/wy13/CR3_choice/wy_rank/scan/rotation_curve.json --oracle outputs/wy13/CR3_choice/oracle/scan/rotation_curve.json --output outputs/wy13/CR3_choice/comparison
```

**UPLOAD:** `CR3_wy13_choice_screen.zip`, made with:

```bat
tar -a -cf CR3_wy13_choice_screen.zip -C outputs/wy13/CR3_choice oracle wy_rank comparison choice_mlp/fit.json choice_mlp/training_history.csv
```

**No heavyweight raw datasets / NN checkpoints are needed for review.** The ZIPs include fitted coefficient arrays and complete curve CSV/JSON, fit quality and gate reports.

## Other models

The same `data.py` supports `C2`, `C3` with `--z-degree 1 --x-degree 2`, `toy_a` / `toy_b` with `--z-degree 2`, and `CR3` with `--z-degree 2`. C2/C3 have an exactly polynomial internal logit and can be tested after C1/CR3 using the same three stage commands; the toy power-law functions are NOT polynomial in Z, so they should be treated as approximation-limited diagnostics (Oracle R² gating applies). For NP extracted latents, the true-parameter oracle is not valid; no deceptive comparison is provided.

## Files

- `basis.py` — training-only polynomial dictionary and analytic Z derivatives
- `fit.py` — link-free rank SVM, link-naive ridge control, isotonic output calibration, optional binary-choice surrogate
- `oracle.py` — oracle-only true-logit polynomial projection, approximation R²
- `scan.py` — fixed-structure SO(2) latent scans, no X derivatives
- `gate2.py` — Oracle-only automatic Gate 2 PASS/FAIL (before WY fit)
- `compare.py` — preset Gate 2/3 pass/fail
- `tests/test_screen.py` — unit checks for rank invariance, C1 axes, nonlinear Z gradient, CR3 valid stimulus processing, binary-label leakage guard

**Methodological limits:** Rank-based fitting gives invariance to common **strictly increasing** output links only when the index model class is adequate, and ties/noise can reduce robustness. The chosen polynomial feature basis is an approximation, not symbolic regression and not full arbitrary nonlinear modules. Optimizing its L1 does not uniquely identify a physical computation graph. No confirmation bias: negative results are a useful screen outcome.
