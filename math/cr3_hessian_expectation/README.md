# CR3 Hessian expectations: analytic versus data-only

Independent experiment in math/cr3_hessian_expectation/. No NP training, no
rotation code or changes to any previously generated files.

## Run

At repository root, with NumPy, SciPy, matplotlib installed:

    conda activate py11
    python np/s1_generate.py --model CR3 --subjects 5000
    python math/cr3_hessian_expectation/run.py

More accurate (slower) prior quadrature:

    python math/cr3_hessian_expectation/run.py --quad-nodes 12 --degree-t 4 --degree-x 3 --ridge-t 0.01 --ridge-x 0.01

Unit tests without requiring the generated dataset:

    python -m unittest discover -s math/cr3_hessian_expectation -p "test_*.py" -v

## Data

Read outputs/choice/CR3/s1_generate/s1_generate__dataset.npz.

- T (n,4): true alpha, loss-aversion lambda, gamma, tau.
- X (500,6): shared 500 trials, always held fixed.
- binary (n,500): actual Bernoulli choices in S1.
- probability (n,500): saved predictive_true.
- logit (n,500): computed once from the CR3 formula because S1 did not
  save pre-sigmoid logits; subsequently passed as numeric Y to the
  data-only method.

Binary is NOT treated as a differentiable random 0/1 function. The
analytic binary comparison is the Hessian of its conditional mean,
E[binary|T,X] = p. The binary data-only regression fits the original
noisy binary Y, not the saved true p.

Hessian variable order, 10 dimensions:

    alpha, lambda_loss, gamma, tau,
    gain_A, loss_A, prob_A, gain_B, loss_B, prob_B

Report theta-theta (4x4), theta-X (4x6), and X-X (6x6).

## Expected Hessian: two different target distributions

Sample expectation, over original 5000 subjects and exact 500 trials:

    (1/N) sum_i (1/500) sum_t H(T_i,X_t)

Prior expectation, integrating only the CR3 population prior:

    integral [ (1/500) sum_t H(theta,X_t) ] pi(theta) dtheta

Prior (independent across four dimensions):

- alpha ~ Beta(12,3)
- lambda ~ Gamma(shape 9, scale 0.22), conditioned on [0.3,5]
- gamma ~ Beta(9,4)
- tau ~ Normal(mean 2.2, sd 0.55), conditioned on [0.5,4]

The prior integral uses tensor-product Gauss-Legendre numerical
quadrature with marginal density weights. It is NOT a closed-form
evaluation. Try 8/10/12 nodes per dimension for convergence. X is NEVER
integrated over a hypothetical continuous stimulus distribution: it
always uses exactly the finite 500-trial average.

## Oracle / analytic branch

oracle.py reproduces the CR3 logit equation exactly:

    w(p,gamma) = sigmoid(gamma*logit(clip(p,1e-12,1-1e-12)))
    V = w(p,gamma)*(gain/100)**alpha
      - lambda*w(1-p,gamma)*(loss/100)**alpha
    eta = tau*(V_A-V_B), choice probability = sigmoid(eta)

Evaluate the full 10x10 H_eta algebraically. Obtain probability Hessian
via the multivariate sigmoid chain rule:

    H_p = p*(1-p) * [H_eta + (1-2*p)*grad_eta*grad_eta.T]

## Data-only branch

data_only.py NEVER imports the oracle function or any code from another
experiment. It is passed only numerical T, X, Y. It uses a tensor-product
polynomial sieve with separate features for T and X, regularized by
two-sided Tikhonov least squares. No 2.5-million-row design matrix is
constructed. All first/second derivatives are exact derivatives of the
fitted polynomial basis; expected first/second moments use factored
feature moments under the T-by-X product measure.

Train/test split is over SUBJECTS, so 500 trial outcomes for a test
subject are never used in training. Holdout RMSE and MAE are reported.

The data-only Hessian is an ESTIMATE under the polynomial sieve
smoothness assumptions, not an exactly identified mathematical object.
A small prediction error does not necessarily imply accurate Hessians.

X has only 500 distinct fixed design points. Continuous X-derivatives
cannot be uniquely identified from those points alone. The fitted H_TX
and H_XX are dependent on the chosen smooth polynomial extension.

## Critical domain issue

Monetary values can be ZERO, while alpha is in (0,1). The original
fractional-power function has no finite amount derivative at zero.
Consequently the true Hessian at those trial points is not a finite
10x10 matrix. Do not interpret a global 10x10 average containing
undefined entries as well-defined.

Every result separates:

- strict_mean: NaN/null whenever an entry is undefined on positive
  target probability mass;
- conditional_mean: average only over points where the analytic entry
  is defined; NOT the unconditional expectation;
- fraction_defined: share of sample/prior mass where defined;
- conditional_rms, conditional_std: summaries on the defined domain;
- model_unmasked_mean: for data-only polynomial alone, a smooth model's
  derivative including zero-amount X; this is NOT a true generator derivative.

The theta-theta block remains defined at zero-amount trials, and is the
primary unconditionally comparable block.

The sigmoid pre-activation and probability Hessians are distinct.
At p=0 or p=1 the exact generator clips probabilities first; the code
respects this piecewise definition except at the clipping knots.

## Output

Saved to outputs/math/cr3_hessian_expectation/:

- summary.json: data hash, assumptions, hyperparameters, quadrature
  normalization errors, holdout accuracy, and per-block comparisons;
- all_matrices.npz: every numeric 10x10 result;
- individual CSV matrices with column/row labels, using
  sample/prior, analytic/data_only, binary/logit/probability, and statistic;
- PNG comparisons of strict Hessian means, theta-X conditional means
  and theta-theta RMS.

Run different --quad-nodes to assess quadrature stability and different
polynomial degrees / ridge parameters to assess estimator sensitivity.
