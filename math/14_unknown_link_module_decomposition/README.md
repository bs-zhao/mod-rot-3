# 14 · Profiled unknown-link sparse module decomposition — C1/C2/C3 **and CR3**

**Independent method screen. Not a rescue of the rejected methods 11–13.** Based on the functional-separability/unknown-link generalized additive modeling literature, with a constrained inverse response transform plus sparse tensor decomposition. The formulation here is an **original screening implementation**; it is **not** Breiman–Friedman ACE, GSAMUL, or an exact replication of their optimization algorithms.

## Scientific question

With fixed observations `W=[Z,X]`, `Y`, can we find a strictly increasing transform `t(Y)` such that the transformed function has few *Z axis × X function module* interactions? Does this select the correct original Z axes in C1, and does it produce meaningful, reproducible rotation preferences in nonlinear CPT **CR3**?

Model class (only a hypothesis, not assumed true):

```
t(Y) ≈ a + f_Z(Z) + f_X(X) + Σ_{i,k} β[i,k] ψ_i(Z) φ_k(X)
```

- `t`: **unknown**, learned from training WY alone, piecewise linear monotone, with positive derivative lower bound and normalized endpoint range. No `logit` or `sigmoid` supplied to the blind method.
- `ψ_i`: polynomial Z basis, degree 1 for C1/C2/C3, degree 2 for CR3. These are *not* the true CPT formula.
- `φ_k`: fixed polynomial X basis, degree 2. Univariate squares share a variable module; cross products are separate X-pair modules. For discrete CR3, features are **only evaluated on real valid sampled lottery rows**: no gradient or perturbation in X is taken.
- The response transform is fitted **once** against the full tensor-sieve column space, before rotating coordinates. An **unpenalized orthogonal projection** (rank-revealing QR) is used to keep the profile fit invariant to rotations of Z's polynomial basis. Monotonicity, noncollapse and smoothness constraints prevent constant-output degeneracy.
- For each fixed Givens rotation, use identical ElasticNet hyperparameters on the same original train rows. Refit **once per rotation**; no hyperparameter selection over angles, no oracle signal given to WY fitting. Validation R² determines whether the fit is adequately comparable to other orientations (within 0.02 of the best); test set is evaluation only.
- For `module_l21`: every nonzero tensor coefficient contributes to a `(Z axis, X module)` interaction. Polynomial Z terms involving two axes allocate squared coefficient energy equally across these axes; then compute the L1 norm of groupwise magnitudes, normalized by their total L2 energy. Also report effective latent partners per X module and the matrix. This **basis-dependent module definition** is not a universal measure of the unknown internal computation graph.

This is substantively different from 13: it **does not use rank SVM, pre-transform labels into pairwise ranks, or analytically rotate a fixed polynomial coefficient matrix**. It profiles a monotone inverse output link via constrained optimization and fits sparse functional components across candidate rotations.

## CR3 limitations — must be visible

CR3 theta is 4D, X is **six lottery attributes** (finite valid stimulus set). The true CPT power and probability-weighting structure is **not generally a degree-2 polynomial of Z and X**; our sieve approximates it. The Oracle stage explicitly tests approximation adequacy and structural informativeness. A failure is a failed **model class + complexity objective**, not proof that unknown-link functional separation or CPT modeling is impossible.

The true coordinates are independently standardized for this screen; orthogonal rotations are of standardized Z. Because CR3 parameter scales differ, this is not a physical-unit rotation. CR3 has no theoretically mandated 0-degree optimum for every complexity definition. Never claim that a CR3 oracle minimum at an arbitrary angle identifies true factors.

`--label prob` means the **simulation-only** condition where each trial's true choice probability is released as observable Y. For `--label bernoulli`, the *blind* stage must first fit a train-only binary-choice probability surrogate (see commands below). The Oracle stage may call true CPT equations only for retrospective screening. NP-inferred Z has no analytic theta oracle: no false oracle comparisons; choose context-only extracted Z and evaluate without the oracle target if that experiment is later defined.

## Preregistered go/no-go gates

**Gate 2 (Oracle, must run first):** fit a sparse functional model on *true internal logit*, but use exactly the same Z/X dictionary and elastic penalty as WY. Require the Oracle winning model's test transformed-target R² ≥ **0.90**, all orientations converged, and improvement in `module_l21` ≥ **3%**. C1/C2/C3 additionally require minimum within **10°** of the original coordinates (modulo 90° signed permutations), and compare zero rotation to **exact** 45° control. CR3 only tests nonflatness across *eligible orientations*, never assumes any specific correct rotation. If Gate 2 FAIL, **stop** and upload the Oracle files.

**Gate 3 (WY, only after Gate 2 passes):** Pearson correlation between the *whole* WY and Oracle continuous rotation curves ≥ **0.90**; selected angle error ≤ **10°** modulo signed permutations; all fits converged. Also store Spearman for diagnosis, but do not use it as a gate because noisy near-ties on largely flat local plateaus can flip ranks without changing the curve geometry. Oracle adequacy is a prerequisite. **These are screening thresholds**, not formal statistical significance. Do not retune an experiment that fails them.

Note that inverse-link identifiability requires enough variation in Y, suitable stimulus overlap and an adequate model class. A near-zero prediction error does **not** establish accurate interaction recovery. The fitted outer link is defined only up to a restricted scale, hence explicit endpoint normalization.

## Windows CMD — from `D:\Projects\mod-rot-3`

Start:

```bat
conda activate py11
git pull
python -m pytest -q math/14_unknown_link_module_decomposition/tests
```

### C1: Oracle Gate 2

Run once if C1 WY splits do not already exist; do not overwrite to change results after screening:

```bat
if not exist "outputs\choice\C1\s1_generate\s1_generate__dataset.npz" python np/s1_generate.py --model C1
python math/11_wy_nonlinear_interactions/data.py --model C1 --label prob --subjects 200 --trials 60 --output outputs/wy14/C1_prob
python math/14_unknown_link_module_decomposition/scan.py --data outputs/wy14/C1_prob --mode oracle --axis-i 1 --axis-j 2 --z-degree 1 --x-degree 2 --step 15 --output outputs/wy14/C1_prob/oracle
python math/14_unknown_link_module_decomposition/gate.py --oracle outputs/wy14/C1_prob/oracle --output outputs/wy14/C1_prob/oracle
```

If `oracle/gate_report.json` has `"gate2": "FAIL"`, STOP. Upload **`C1_wy14_oracle.zip`**:

```bat
tar -a -cf C1_wy14_oracle.zip -C outputs/wy14/C1_prob oracle
```

If PASS, WY branch:

```bat
python math/14_unknown_link_module_decomposition/scan.py --data outputs/wy14/C1_prob --mode wy --axis-i 1 --axis-j 2 --z-degree 1 --x-degree 2 --step 15 --output outputs/wy14/C1_prob/wy
python math/14_unknown_link_module_decomposition/gate.py --oracle outputs/wy14/C1_prob/oracle --estimated outputs/wy14/C1_prob/wy --output outputs/wy14/C1_prob/comparison
```

**UPLOAD `C1_wy14_screen.zip`**:

```bat
tar -a -cf C1_wy14_screen.zip -C outputs/wy14/C1_prob oracle wy comparison
```

### CR3: Oracle Gate 2 — required independently of C1 outcome

```bat
if not exist "outputs\choice\CR3\s1_generate\s1_generate__dataset.npz" python np/s1_generate.py --model CR3
python math/11_wy_nonlinear_interactions/data.py --model CR3 --label prob --subjects 260 --trials 90 --output outputs/wy14/CR3_prob
python math/14_unknown_link_module_decomposition/scan.py --data outputs/wy14/CR3_prob --mode oracle --axis-i 0 --axis-j 2 --z-degree 2 --x-degree 2 --step 15 --output outputs/wy14/CR3_prob/oracle
python math/14_unknown_link_module_decomposition/gate.py --oracle outputs/wy14/CR3_prob/oracle --output outputs/wy14/CR3_prob/oracle
```

If FAIL, STOP. **UPLOAD `CR3_wy14_oracle.zip`**:

```bat
tar -a -cf CR3_wy14_oracle.zip -C outputs/wy14/CR3_prob oracle
```

If PASS, WY branch:

```bat
python math/14_unknown_link_module_decomposition/scan.py --data outputs/wy14/CR3_prob --mode wy --axis-i 0 --axis-j 2 --z-degree 2 --x-degree 2 --step 15 --output outputs/wy14/CR3_prob/wy
python math/14_unknown_link_module_decomposition/gate.py --oracle outputs/wy14/CR3_prob/oracle --estimated outputs/wy14/CR3_prob/wy --output outputs/wy14/CR3_prob/comparison
```

**UPLOAD `CR3_wy14_screen.zip`**:

```bat
tar -a -cf CR3_wy14_screen.zip -C outputs/wy14/CR3_prob oracle wy comparison
```

### CR3: real Bernoulli choice condition — only if the CR3 probability test passed

```bat
python math/11_wy_nonlinear_interactions/data.py --model CR3 --label bernoulli --subjects 260 --trials 90 --output outputs/wy14/CR3_choice
python math/11_wy_nonlinear_interactions/models.py --data outputs/wy14/CR3_choice --method mlp --epochs 200 --output outputs/wy14/CR3_choice/choice_mlp
python math/14_unknown_link_module_decomposition/scan.py --data outputs/wy14/CR3_choice --mode oracle --axis-i 0 --axis-j 2 --z-degree 2 --x-degree 2 --step 15 --output outputs/wy14/CR3_choice/oracle
python math/14_unknown_link_module_decomposition/gate.py --oracle outputs/wy14/CR3_choice/oracle --output outputs/wy14/CR3_choice/oracle
```

If Gate 2 passes:

```bat
python math/14_unknown_link_module_decomposition/scan.py --data outputs/wy14/CR3_choice --mode wy --source-fit outputs/wy14/CR3_choice/choice_mlp --axis-i 0 --axis-j 2 --z-degree 2 --x-degree 2 --step 15 --output outputs/wy14/CR3_choice/wy
python math/14_unknown_link_module_decomposition/gate.py --oracle outputs/wy14/CR3_choice/oracle --estimated outputs/wy14/CR3_choice/wy --output outputs/wy14/CR3_choice/comparison
tar -a -cf CR3_wy14_choice_screen.zip -C outputs/wy14/CR3_choice oracle wy comparison choice_mlp/fit.json choice_mlp/training_history.csv
```

**UPLOAD `CR3_wy14_choice_screen.zip`**; else upload `CR3_wy14_choice_oracle.zip` created by:

```bat
tar -a -cf CR3_wy14_choice_oracle.zip -C outputs/wy14/CR3_choice oracle
```

No huge neural-network checkpoint or raw WY is needed in the upload.

## Output files

Each `scan.py` produces `rotation_curve.json`, `rotation_curve.csv`, `rotation_curves.png`, and `summary.json`. The JSON includes every rotation's test and validation R², convergence flag, complete axis × stimulus module matrix, output transform's monotone knots/slopes (WY only), and chosen angle. `gate.py` writes `gate_report.json`, with all thresholds and PASS/FAIL.

### Research sources (methodological motivation, not claimed exact reproductions)

- Breiman & Friedman (1985), *Estimating optimal transformations for multiple regression and correlation* (ACE).
- Horowitz & Mammen (2007), *Rate-optimal estimation for a general class of nonparametric regression models with unknown link functions*.
- Yuan et al. (2024), *Generalized Sparse Additive Model with Unknown Link Function* (GSAMUL).
- Jacho-Chávez, Lewbel & Linton (2010), *Identification and Nonparametric Estimation of a Transformed Additively Separable Model*.
- Ba, Melnyk & Wald (2024), *Sparse additive function decompositions facing basis transforms*.
