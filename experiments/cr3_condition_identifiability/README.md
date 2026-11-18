# CR3 stratified identifiability and analytic rotation — A/B/C

Independent **analytic-only** experiment. It **does not train a neural network**,
load/modify a frozen checkpoint, change S0/S1, or import any previous rotation
package. All new code lives in `experiments/cr3_condition_identifiability/`;
all new results in `outputs/cr3_condition_identifiability/<tag>/`.

**Question:** The exact CPT/CR3 likelihood can use structurally special trials
to distinguish alpha, lambda, gamma, tau. Will derivatives' conditional
geometric/simplicity criteria likewise prefer the true parameter axes?
Separately, do the zero-amount singularities invalidate X derivatives,
even though the same trials remain valid and useful for likelihood/parameter
derivatives?

## Source of truth, geometry, scaling

Read-only existing artifacts:

- `outputs/choice/CR3/s0_design/CR3_4P_fixed500.npz`: exact 500×6
  immutable bank, `type_codes`, `type_names`.
- `outputs/choice/CR3/s1_generate/s1_generate__dataset.npz`: true `parameters`
  for subjects, used for the oracle-only study; no NP z or decoder.
- `three_gaps/common/analytic.py:eta_cr3` and independent self-check
  against `np/reference/s0_cr2_design_recovery.py:logits`.

Original trial types are **disjoint**: gain-only 60, gamma-neutral 60,
loss-only 40, general 340, based on bank's native labels (do not reconstruct
these labels from overlap-prone heuristics). X physical columns are
`(gain_A, loss_A, p_A, gain_B, loss_B, p_B)`.
`p` is probability of gain; loss has probability `1-p`.

Function (raw theta):

```
V(g,l,p;alpha,lambda,gamma) =
  w(p,gamma) * (g/100)^alpha
  - lambda * w(1-p,gamma) * (l/100)^alpha

w(p,gamma) = sigmoid(gamma * logit(p))
eta = tau * (V_A - V_B)
p_choice = sigmoid(eta)
```

For derivatives, define **true-param standardized**
`t=(theta-mu_theta)/sd_theta` using original dataset parameters; gradients,
Hessians and rotated axes U are in these same standardized units. No
supervised `theta~AZ` proxy is used, so this stage isolates the conceptual
identifiability question from imperfect NN decoding/mapping.

Physical boundary rules and finite differences always use **raw** X.
`--x-scale standardized` (default) only multiplies X Jacobian columns
by the original bank's per-column standard deviation so cross-X energy
comparisons use common coordinates. In the CLI, this is **analysis
coordinate rescaling**, not NN training normalization. With
`--x-scale raw` the output uses physical units. Do not confuse these
two derivative norms.

We consider logit and probability separately. Fisher information always
uses logit gradients, `p(1-p) grad_t(eta) grad_t(eta)^T`, including when
analyzing probability derivatives.

## Stage A: parameter-Fisher complementarity

For each condition s, compute

```
F_s = E_{t,X|s}[p(1-p) grad_t(eta) grad_t(eta)^T].
F_empirical = sum_s (n_s / 500) F_s
F_balanced  = sum_s (1/4) F_s
```

Record matrices, eigenvalues/ranks, full-bank mixtures, and `G_cov_s`.
The generator itself supplies two exact structural nulls:

- gain-only: partial eta / partial lambda = 0
- gamma-neutral (`p_A=p_B=.5`): partial eta / partial gamma = 0

Loss-only separately confounds the product lambda·tau **at fixed theta**;
population-averaged Fisher need not retain that null because theta varies
between subjects. Likelihood can be identifiable from complementary
conditions even if a single condition is not.

**Rotation losses (lower=better)**: conditional joint off-diagonal Fisher
energy, plus a separate `two_condition_null` objective. The latter
finds the minimum diagonal Fisher energy at two **distinct** rotated axes
for gain-only and gamma-neutral. It uses condition labels, but never uses
the true parameter-name-to-axis correspondence. These are *structural*
biases, not maximized Fisher determinant: determinant, eigenvalues, rank
are invariant to orthogonal R and cannot choose R.

## Stage B: U-only derivative simplicity

On original 500 rows, including zero amounts:

```
G(t,x) = grad_t f
H(t,x) = d2f/dt dt
```

Parameter derivatives remain well-defined at zero amounts for
`0<alpha<1`: `0^alpha=0`, `d/dalpha 0^alpha=0`.
For each category preserve `E(GG^T)` and the **fourth-order covariance**
`E(H_ab H_cd)`. At any R, derive `E[(RHR^T)_ij²]` *exactly from
these fixed moments*; no re-evaluation of the generating function and no
derivative-approximation error during rotation.

Loss families: off-diagonal Hessian squared-energy proportion, tail
energy after four largest Hessian matrix elements, and gradient
energy outside the two largest rotated U coordinates. Compute by
condition and by empirical/balanced mixtures.

Important: these are **hypotheses about simplicity**, not a theorem of
true-axis recovery. The CPT formula has real multiplicative interactions.

## Stage C: X-domain-aware interaction

At each unchanged bank row, record

```
UX(t,x) = d2f/dt dx
```

Two separate masks:

1. `ambient_valid`: all positive amounts or interior probabilities
   `0<p<1` are valid for that particular X coordinate. A zero amount
   or endpoint p is excluded; never claim the zero from a `torch.where`
   branch is a valid classic derivative.
2. `tangent_valid`: ambient valid **and** a direction varying within the
   trial's declared condition. The two probabilities of gamma-neutral are
   omitted here, even though ambient differentiation at p=.5 is finite.
   Fixed-zero loss/gain coordinates are omitted in gain-/loss-only.

Use exact conditional covariance
`K_sj=E[UX_{:j} UX_{:j}^T | s, valid_j]` and store separate
`support_sj=P(valid_j|s)`. NEVER replace undefined coordinates with
made-up derivative zeros. Column-weighted rotation losses compare
the proportion of U-axis energy outside the single largest axis per X
column (`C_*_top1`).

In addition create **finite-scale** X perturbations, with raw physical
steps `--amount-step 5` and `--probability-step .05`:

```
Q_{ij}^h(t,x) = [partial_t_i f(t,x_plus) -
                 partial_t_i f(t,x_minus)] / (x_plus_j - x_minus_j)
```

Choose two-sided differences when probes stay in original bank's
coordinate range (amount 0–200, probability 0–1), and a one-sided
difference otherwise. This does **NOT** estimate the nonexistent
classic derivative at a zero amount. Keep three separate classes:

- `finite_tangent`: condition-preserving X coordinates;
- `finite_boundary`: at zero amounts or probability endpoints, a
  physically legal one-sided intervention that may exit the trial family;
- `finite_cross`: a smooth coordinate that nonetheless exits the
  condition by changing a fixed attribute (e.g. p in gamma-neutral).

These are finite **contrasts**, not derivative limits. Raw
steps, widths, masks, condition shares, and both unmodified X/theta are
retained in `fields.npz`. The NN has not been evaluated off-support
in this analytic-only benchmark.

## Search protocol and honesty safeguards

All losses use `U=R t`, with `R in SO(4)`. Transform gradients/Hessians:

```
grad_U f = R grad_t f
H_UU = R H_tt R^T
H_UX = R H_tX
F_U = R F_t R^T
```

- **Diagnostic**: identity true axes and all 6 pairwise planes at
  angles 15,30,45,60,75,90° (90° is signed-permutation equivalent).
- **Oracle-blind search**: independent Haar random SO(4) candidates
  plus BFGS optimization from random six-angle seeds. **Identity
  and oracle-centered plane sweeps are excluded from the set
  considered as searched optima and from optimizer initialization.**
  The metric computation does not read theta axis labels.
- **Afterwards**, compare searched optima to known true theta axes
  up to global signed permutation (Hungarian matching). Report average
  |cos| and maximum aligned axis angle. Also print identity loss so one
  can tell whether a candidate beats the real axes.

The search does not guarantee a global optimum, especially for
non-smooth top-k/min-diagonal objectives. Test robustness across
random seeds, optimizer restarts, bank mixtures, and sample sizes.

## Commands — Windows conda py11

At project root `D:\Projects\rotation\mod-rot-3`:

```cmd
conda activate py11
git pull origin main
python -m pytest experiments/cr3_condition_identifiability/tests -q
```

**First smoke (2–4 subjects, all four conditions, no optimizer):**

```cmd
python -m experiments.cr3_condition_identifiability.run --stage all --tag smoke --subjects 4 --stimuli 32 --channels logit --random-rotations 12 --no-optimize --batch 32
```

**Main analytic-only experiment (all 500 unchanged X, two channels):**

```cmd
python -m experiments.cr3_condition_identifiability.run --stage all --tag analytic_cr3_64 --subjects 64 --stimuli 500 --channels logit prob --random-rotations 128 --restarts 3 --maxiter 120 --batch 128
```

**Separate phases without recomputing analytic derivatives:**

```cmd
python -m experiments.cr3_condition_identifiability.run --stage precheck --tag analytic_cr3_64 --subjects 64 --stimuli 500
python -m experiments.cr3_condition_identifiability.run --stage derive --tag analytic_cr3_64 --subjects 64 --stimuli 500 --channels logit prob
python -m experiments.cr3_condition_identifiability.run --stage search --tag analytic_cr3_64 --channels logit prob --random-rotations 128 --restarts 3 --maxiter 120
```

To repeat a stage with changed number of subjects, bank selection, or
physical finite-scale steps, use a **new tag**. `--force` explicitly
recomputes stored derivatives for the same tag; no old experiment is edited.
The `--stage search` step reads its own saved `moments.npz` only.

## Output contract

```
outputs/cr3_condition_identifiability/<tag>/
  config.json
  precheck.json
  REPORT.md
  logit/  (and/or prob/)
    fields.npz                # full derivatives, masks, unchanged X
    moments.npz               # sufficient moment tensors for fast R search
    diagnostics.json          # four condition Fisher and support/rank
    rotations_all_scores.csv  # 6 plane landscapes and all SO4 candidates
    best_rotations.json       # per objective/mixture best R and alignment
    figures/                  # pairwise 0..90 landscapes
```

## Interpretation boundaries

- **This study does NOT yet evaluate neural-network derivative fidelity**.
  It first establishes whether analytic losses can recover truth at all.
- A trained NN may fit a discrete 500-item likelihood yet learn a smooth
  extension across the non-differentiable zero-amount boundary. Its value
  likelihood does not identify X derivatives between observed stimuli.
- A Fisher rank result concerns known theta, *not* choosing latent Z axes.
  Rotating theta and appropriately transforming an unrestricted decoder
  leaves the likelihood unchanged.
- Per-condition score minima may disagree. Aggregation is a defined
  empirical/balanced objective, never a proof of correct true axes.
- Discarding all boundary stimuli and silently averaging only the
  112 fully smooth rows changes the design distribution; this suite
  keeps all 500 and handles validity per X column.
- Verify the analytical selfcheck and structural-zero diagnostics before
  interpreting any minimum. No model training is performed.
