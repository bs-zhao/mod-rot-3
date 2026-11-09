# Sequential three-gap experiments

All code and generated artifacts for this series stay inside this directory.
Original NP weights, data, Z, and other experiments are read-only.

Each `run.py` invocation evaluates ONE loss, ONE model, ONE response channel,
and ONE source/estimator. There is no all-method rotation runner.

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
claim of inference from the original observational WY table. Binary
observations are rejected until a Bernoulli estimator is implemented.

Example (from repository root, py11 environment):

```cmd
python research/three_gap_series/run.py --model CR3 --beta 0.00005 --zdim 4 --pca 0 --metric uu_self_variance --channel logit --source formula --samples 384 --starts 8
```

All results, seeds, sampled row IDs, losses, multistart solutions and plane
curves are saved under `results/<round>/<model_beta_zdim_pca>/...`.
TEST T is read only after the decoder rotation is saved to disk. No loss
weights or methods may be chosen using TEST results. Further experiments
must distinguish exploratory screening from independent confirmation.
