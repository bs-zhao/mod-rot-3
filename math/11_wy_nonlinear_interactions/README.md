# 11 · WY-only nonlinear interactions — independent benchmark

**Scope.** Estimate nonlinear interactions and coordinate-dependent structural complexity from paired samples `W=[Z,X],Y` with *no analytical f, gradients, logits, theta labels, or generator internals available to the estimator*. Models: `toy_a`, `toy_b`, `C1`, `C2`, `C3`, `CR3`. Every operation is a separate script; none changes the existing NP pipeline. Run commands from repository root in `conda activate py11`.

## Implementation and honest method inventory

| Implemented predictor | What it actually is | Install |
|---|---|---|
| `mlp` | 3-layer SiLU PyTorch MLP; classification uses BCE-with-logits, regression uses MSE | root requirements |
| `spline_add` | univariate spline additive main effects + regularized regression | root requirements |
| `spline_pair` | tensor-product spline main/pairwise interactions + regularized regression | root requirements |
| `ebm` | real InterpretML Explainable Boosting Machine with automated pairwise interactions | `pip install interpret` |

Implemented **detectors** applied to fitted predictors: `fd` (RMS mixed central difference), `hessian` (autodiff Hessian, `mlp` only), `friedman` (centered two-way partial-dependence H), `anova` (Monte Carlo product-intervention functional ANOVA, up to user-selected order). These are different *estimands*, so compare with an oracle using the same detector. `score.py` refuses cross-estimand comparisons.

**Not implemented as papers' algorithms:** SKIM-FA, OAK, GAMI-Net, and Diamond. They require verified dedicated integrations/dependencies; `spline_pair` is **not** SKIM-FA, `spline_add` is **not** OAK, and a plain MLP is **not** GAMI-Net. Diamond FDR is **not** claimed. This module is an executable screening harness for the four implementations above; the other four remain explicitly out of scope for this commit.

## Method isolation, evaluation targets, and validity

- `data.py` is the *sole preparation stage*. It imports simulation data, creates subject-disjoint splits and labels, and writes three method-facing files `train.npz`, `val.npz`, `test.npz` containing `W,y,groups`. No true parameter *names* or formula is provided. True parameter **values** are columns of W only in true-coordinate conditions, as specified by the experiment.
- `oracle_reference.npz` contains test truth and is **never opened by** `models.py`, `interactions.py`, or blind `rotation.py`. Only `oracle.py` and its scoring interface may access true formulas/values. If using NP Z, an analytic theta-coordinate oracle is **not valid**; `oracle.py` rejects it.
- `label=logit|prob|softplus|bernoulli` provides four output scales. `logit` is a simulation label, not observed choice data. `bernoulli` uses the recorded 0/1 responses from repository generators (or simulated draws for toy).
- Correlated inputs invalidate classical independent-input Sobol interpretations. `anova` estimates an **interventional/product-reference** decomposition with finite MC error, not a guaranteed orthogonal decomposition for dependent observational inputs.
- CR3 X consists of six discrete lottery fields, fixed 500 valid stimuli, including zero amounts and probability boundaries. Naively perturbing X is off support and may be non-differentiable. `fd`/`hessian`/`friedman`/`anova` thus mark entries involving CR3 X as NaN. Instead they report `valid_stimulus_z_modulation`: variation of Z-sensitivity across *pairs of genuine observed lottery vectors*. This is a valid-bank, pooled ZX measure **not** a per-X-coordinate interaction matrix. It does not identify individual X edges.
- For binary Y, detected interactions refer to **probability scale**; oracle can separately evaluate `logit`-scale structure. Only an identified link permits interpreting the internal scale. Predictive fit quality does not imply consistency of first/second derivatives.
- Interaction scores are global RMS or PD scores. `zx_effective_degree` is the sum over X columns of entropic effective Z partner count, well-defined when ZX scores exist. It is **not** a guaranteed unique rotation objective. Do not optimize `Frobenius(H_ZX)` under orthogonal rotations; it is invariant.
- `score.py` uses a relative oracle threshold (default 5% of max) **only for retrospective scoring**. Top-K precision uses oracle K and must never be reported as a deployable interaction classifier. Spearman, RMSE and zero-edge cases remain separate.
- All `fit` and detector scripts are real WY-only methods, but different fits may need tuning. Use independent seeds and held-out subjects. Never tune based on oracle results.

## Quick start: toy B, probability scale

```bat
conda activate py11
pip install -r requirements.txt
python math/11_wy_nonlinear_interactions/data.py --model toy_b --label prob --subjects 180 --trials 40 --output outputs/wybench/toy_b_prob
python math/11_wy_nonlinear_interactions/models.py --data outputs/wybench/toy_b_prob --method mlp --epochs 200 --output outputs/wybench/toy_b_prob/mlp
python math/11_wy_nonlinear_interactions/interactions.py --data outputs/wybench/toy_b_prob --fit outputs/wybench/toy_b_prob/mlp --detector hessian --n-eval 96 --output outputs/wybench/toy_b_prob/mlp/hessian
python math/11_wy_nonlinear_interactions/interactions.py --data outputs/wybench/toy_b_prob --fit outputs/wybench/toy_b_prob/mlp --detector fd --n-eval 96 --output outputs/wybench/toy_b_prob/mlp/fd
python math/11_wy_nonlinear_interactions/oracle.py --data outputs/wybench/toy_b_prob --scale prob --detector fd --n-eval 96 --output outputs/wybench/toy_b_prob/oracle_prob_fd
python math/11_wy_nonlinear_interactions/score.py --estimated outputs/wybench/toy_b_prob/mlp/fd --oracle outputs/wybench/toy_b_prob/oracle_prob_fd --output outputs/wybench/toy_b_prob/score
```

**Note:** The Hessian detector and FD detector both target mixed curvature, but their finite-step estimates are not numerically identical. Compare FD to FD for a fair score.

## Rotation experiment (fixed learned predictor; no retraining)

```bat
python math/11_wy_nonlinear_interactions/rotation.py --data outputs/wybench/toy_b_prob --fit outputs/wybench/toy_b_prob/mlp --axis-i 0 --axis-j 1 --step 5 --detector fd --output outputs/wybench/toy_b_prob/rotation_wy
python math/11_wy_nonlinear_interactions/rotation.py --data outputs/wybench/toy_b_prob --oracle-scale prob --axis-i 0 --axis-j 1 --step 5 --detector fd --output outputs/wybench/toy_b_prob/rotation_oracle
python math/11_wy_nonlinear_interactions/compare_curves.py --estimated outputs/wybench/toy_b_prob/rotation_wy/rotation_curve.json --oracle outputs/wybench/toy_b_prob/rotation_oracle/rotation_curve.json --output outputs/wybench/toy_b_prob/rotation_compare
```

Rotation holds the fit **fixed** and uses the exact inverse coordinate transform. This tests *structural estimation*, not retraining variance. An additional separate experiment could retrain each method at each angle for model-selection comparisons. For Toy B (chosen Givens convention) expected structural optimum includes +45 degrees up to sign/permutation equivalences; the Oracle must first be checked to prefer that direction under the chosen metric.

For random multidimensional SO(d) controls: add `--random-rotations 40`. For higher-order ANOVA: `--detector anova --order 3` on `interactions.py`; Monte Carlo error is expected.

## C1/C2/C3/CR3: reuse original data

For C1/C2/C3/CR3, first prepare the original data with the repository's existing `np/s1_generate.py --model CR3` sequence (CR3 also requires its precheck/design prerequisites). We do **not** silently regenerate or overwrite old datasets. Example:

```bat
python math/11_wy_nonlinear_interactions/data.py --model CR3 --label bernoulli --subjects 500 --trials 200 --output outputs/wybench/CR3_choice
python math/11_wy_nonlinear_interactions/models.py --data outputs/wybench/CR3_choice --method mlp --max-train 30000 --output outputs/wybench/CR3_choice/mlp
python math/11_wy_nonlinear_interactions/interactions.py --data outputs/wybench/CR3_choice --fit outputs/wybench/CR3_choice/mlp --detector fd --n-eval 100 --output outputs/wybench/CR3_choice/mlp/fd
python math/11_wy_nonlinear_interactions/oracle.py --data outputs/wybench/CR3_choice --detector fd --scale prob --output outputs/wybench/CR3_choice/oracle
python math/11_wy_nonlinear_interactions/rotation.py --data outputs/wybench/CR3_choice --fit outputs/wybench/CR3_choice/mlp --axis-i 1 --axis-j 3 --step 15 --output outputs/wybench/CR3_choice/rot_lambda_tau
```

CR3 rotation reports **valid-bank stimulus modulation**, because per-coordinate ZX-Hessian is intentionally unavailable. Its theta priors are nonisotropic, so rotating Z requires careful semantic interpretation.

## Frozen NP Z: context-only extraction to prevent leakage

The existing `np/s5_extract_z.py` uses **all observed trials** to encode every person's Z. Its result would leak target labels into Z for a same-trial evaluation. The independent alternative is:

```bat
python math/11_wy_nonlinear_interactions/extract_context_z.py --model CR3 --dataset outputs/choice/CR3/s1_generate/s1_generate__dataset.npz --checkpoint PATH_TO_FROZEN_NP_CHECKPOINT.pt --context-fraction 0.5 --output outputs/wybench/CR3_context_z.npz
python math/11_wy_nonlinear_interactions/data.py --model CR3 --label bernoulli --latent-file outputs/wybench/CR3_context_z.npz --subjects 500 --trials 150 --output outputs/wybench/CR3_npz_choice
```

Extraction writes an explicit `target_trial_mask`; the data loader selects only target trials excluded from context. Checkpoint and data hashes are verified. Do not reuse the all-trial s5 Z in a clean evaluation. The `--allow-leaky-latents` override exists only for a clearly labeled diagnostic and must never be pooled into clean results.

## Outputs

Each method has its own `fit.json`, model artifact (`model.pt` or `model.pkl`), `interaction_matrix.csv`, `interaction_matrix.png`, `interaction.json`; rotation outputs `rotation_curve.json` and plots, score outputs `score.json`, and comparisons output `comparison.json`/plot. Run each method independently, not a giant run-all script.

Run tests: `python -m pytest -q math/11_wy_nonlinear_interactions/tests`.

## Method reference links

- SKIM-FA: https://www.jmlr.org/papers/v24/21-1403.html (not implemented here)
- OAK: https://proceedings.mlr.press/v162/lu22b.html (not implemented here)
- GAMI-Net: https://doi.org/10.1016/j.patcog.2021.108192 (not implemented here)
- Diamond: https://www.nature.com/articles/s42256-025-01086-8 (not implemented here)
- EBM documentation: https://interpret.ml/docs/ebm.html (optional integration implemented)
