# Sequential three-gap experiments

**Scope correction (2026-11-09):** The earlier transport result below is an
OUT-OF-SCOPE CONTROL, not completion of the requested Hessian research.
The active study is [Hessian-only sequential search](HESSIAN_SEARCH.md).
It requires the same Hessian metric and hyperparameter rules for all models,
generic WY inference, and explicit G1/G2/G3 failure records.

**Current priority:** [single Hessian metrics](SINGLE_METRIC_STUDY.md);
combinations and weight tuning are paused. See the
[single-metric results](results/single_metric_study/REPORT.md).

**Earlier numerical result, outside the requested scope:** `transport_dual_cosine` with structured
numerical-WY estimators meets the requested mean-correlation gap <= 0.01
against each model's supervised reference on four fresh cohorts of 2,000
subjects. See [the final report](results/final/REPORT.md), including the
stimulus priors, channel restrictions, negative results and reproducible commands.
This uses first-order functional transports, not a pure Hessian loss.
`make_report.py` and its older figures describe an earlier exploratory snapshot;
use `final_report.py` for the frozen final protocol.

All code and generated artifacts for this series stay inside this directory.
Original NP weights, data, Z, and other experiments are read-only.

Each `run.py` invocation evaluates ONE loss, ONE model, ONE response channel,
and ONE source/estimator. The atlas scores independent single objectives on
shared rotations; each subsequent optimization still evaluates only one loss.

Required flags: `--model --beta --zdim --pca`. PCA is TRAIN-only, without
whitening; currently only equal-dimensional invertible PCA is supported.
All checkpoint, dataset and latent hashes, context mode and subject splits
are checked. Optimizer input contains no true parameters on the decoder path.

Round 1 compares 12 predefined losses with shared field samples, 37-point
scans in every coordinate plane and joint multistart SO(d) optimization.
Every loss is independently audited for arbitrary post-rotation signed
scaling and permutation invariance. Analytic success requires the best
found joint solution to have minimum axis cosine > .995, no loss below
the true axis by more than 1e-6*(1+|L_true|), and a nonflat curve in every
plane. This is a numerical screen, not a proof of global minimality.

The signature losses explicitly augment Hessians with mean gradients and
third derivatives H_zxx. They are NOT pure second-order Hessian losses.
Mixed-third loss uses H_zzx. Self variance is variance over the stated joint
empirical Q, NOT conditional between-subject variance from historical math/07.

Analytic formula implementation follows the two research READMEs. CR3
H_zx statistics use the positive-money, interior-probability part of the
actual design; the retained domain fraction is recorded. C1-C3 use actual
subject/trial rows. All standardizers use TRAIN only, including candidate
axis SD after each rotation. Neither raw Z nor PCA is whitened.

`--estimator finite_difference` is an E09 QUERY-ACCESS numeric benchmark:
central differences query new point/value pairs. It is explicitly NOT a
claim of inference from the original observational WY table.

`--estimator polynomial` is an original-row WY total-degree sieve. Its
weighted polynomial basis and ridge norm are rotationally invariant (tested),
and a hidden rotation is applied before fitting. The estimator receives
numerical input/output arrays only. Binary observations use Bernoulli
likelihood, not differentiation of draws. TRAIN and VALIDATION subjects are
disjoint. `--degree`, `--ridge`, `--train-rows`, and `--validation-rows`
are explicit estimator settings.

`--estimator stimulus` is a CR3 latent-polynomial x fixed-stimulus-indicator
sieve. It provides latent gradients/Hessians only, no continuous X derivatives.
The actual shared 500-row bank is checked for every selected subject.
`--train-subjects` and `--validation-subjects` control grouped sample sizes.

`--structure-link log_odds` is an ADDITIONAL canonical-logit structure
condition: probability observations are mapped through logit, and binary
observations fit a Bernoulli sieve whose latent log odds are differentiated.
This must NEVER be reported as successful recovery using the native
probability Hessian. Native-p and log-odds conditions are distinct.

`signature_plain_l1` uses the L1 amplitude of the mean gradient, mean ZX
Hessian, and mean ZXX derivative signature, after candidate-axis and fixed-X
standardization. It deliberately does not give tiny noisy columns unit weight.
It is amplitude sparsity, not a literal graph-edge count. It includes F1/F3
information and is not a pure second-order Hessian metric.

`conditional_ratios` and its bilinear variants are CR3-design-specific
conditional absence / self-linearity candidates. `--subject-mean 1` means
signed derivative fields are averaged over subjects BEFORE squaring or
aggregating over stimuli: a distinct M1 estimand from local energy averaging.
`conditional_anchor` is an explicitly model-informed, OUT-OF-DESIGN query
control using neutral amounts/probabilities; some lotteries are dominated.
Its results are not original-table or generic model-free results.

The `combo_*` catalogue combines normalized UX soft support with normalized
UU diagonal soft support, using RMS or SD aggregation and a predefined
threshold/weight grid. Each setting is one separately invoked experiment.
`ux_rms_*`, `ux_sd_*`, and `ux_absmean_*` compare amplitude and concentration
losses with the same derivative fields. Selection on analytic simulations
is exploratory; a new subject cohort is reserved for confirmation.

Example (from repository root, py11 environment):

```cmd
python research/three_gap_series/run.py --model CR3 --beta 0.00005 --zdim 4 --pca 0 --metric uu_self_variance --channel logit --source formula --samples 384 --starts 8
```

All results, seeds, sampled row IDs, losses, multistart solutions and plane
curves are saved under `results/<round>/<model_beta_zdim_pca>/...`.
TEST T is read only after the decoder rotation is saved to disk. No loss
weights or methods may be chosen using TEST results. Further experiments
must distinguish exploratory screening from independent confirmation.

`fresh_cohort.py` confirms a frozen result on newly generated independent
subjects and choices, using the original frozen encoder with all trials.
It records the rotation-summary hash before drawing new T and fits nothing.
C1-C3 confirmation uses 500 trials per new subject; this differs from the
original variable 300-500 trial cohort. CR3 keeps the actual 500-stimulus bank.

`oracle_geometry.py` is a separate supervised TRAIN-only R+D MSE / OLS
diagnostic, never imported by discovery. Its multistart R+D solution is a
numerical reference, not a proof of the exact global optimum.

Small reviewable JSON/CSV/PNG results are versioned. Large field caches and
redundant `rotation.npy` files remain local; every R is also in summary.json.
Compact configuration hashes avoid Windows path-length limits. All flags,
source hashes, code hashes and, in newer runs, M1 matrices are recorded.

Validation:

```cmd
python -m unittest discover -s research/three_gap_series -p test_estimators.py -v
python research/three_gap_series/validate.py --model CR3 --beta 0.00005 --zdim 4 --pca 0
```
