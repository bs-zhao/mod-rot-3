# CR3: three independent Hessian estimation algorithms

**One self-contained experiment** in `math/cr3_hessian_estimators/`. This folder
imports no Python modules from any other `math/` experiment or rotation code.
It reads only the saved CR3 S1 dataset, not checkpoints or learned representations.
The methods in `estimators.py` do **not** import the CR3 analytic equation.
The independent `reference.py` is used only by the benchmark runner to create
an analytic gold standard and to construct S1's unstored logit channel.

## Commands (Windows CMD, repo root)

```cmd
conda activate py11
git pull origin main
python -m unittest discover -s math/cr3_hessian_estimators -p "test_*.py" -v
python math/cr3_hessian_estimators/run.py
```

Data must already exist from:

```cmd
python np/s1_generate.py --model CR3 --subjects 5000
```

To run methods individually (outputs always in a single experiment folder):

```cmd
python math/cr3_hessian_estimators/run.py --methods krr
python math/cr3_hessian_estimators/run.py --methods spline
python math/cr3_hessian_estimators/run.py --methods stein
```

Tuning examples:

```cmd
python math/cr3_hessian_estimators/run.py --nystrom-rank 384 --spline-basis 5 --quad-nodes 10
python math/cr3_hessian_estimators/run.py --quad-nodes 12 --spline-basis 6
```

Requires NumPy, SciPy and matplotlib; no neural network or PyTorch.

## Mathematically well-defined target

True parameters T have 4 dimensions (`alpha`, `lambda`, `gamma`, `tau`).
The shared fixed bank contains exactly 500 six-dimensional stimuli X.
The per-trial function is either (i) pre-sigmoid logit eta(T,X), or
(ii) sigmoid probability p(T,X). The Bernoulli binary response targets
its *conditional mean* p(T,X), rather than a nonexistent pathwise Hessian
of a random binary sample.

Define the **stimulus-averaged conditional mean** as

    g(T) = (1/500) sum_{t=1}^{500} f(T,X_t).

By linearity of finite sums and differentiation,

    Hessian_T[g(T)] = (1/500) sum_t Hessian_T[f(T,X_t)].

Thus, for estimating the 4x4 *mean* TT Hessian, one may regress the
per-subject mean response `Ybar_i=(1/500)sum_t Y_it` on the 5000 true
parameter vectors T_i. This is a 4D, not 10D, nonparametric problem.
Crucially, **this does not recover trialwise Hessian RMS, Hessian TX,
or Hessian XX**: averaging destroys trialwise variation.

For each channel, compare:

    H_sample = (1/N) sum_i Hessian_T[g(T_i)]
    H_prior  = integral Hessian_T[g(t)] pi_CR3(t) dt

`H_prior` uses independent CR3 Beta / truncated Gamma / Beta / truncated
Normal priors and Gauss-Legendre integration, while the X bank is fixed.
Increase `--quad-nodes` to assess convergence (8 nodes per dimension is
only an initial, non-exact numerical approximation). The analytic oracle
follows the original CR3 generator's *separately clipped* gain and loss
probability weighting at p=0 and p=1.

The binary data-only regression fits the **actual 0/1 trials averaged over
500**, and never substitutes true p. The probability branch fits the saved
true p averaged over 500. S1 does not store eta: it is computed using the
true formula once as response *data*, then passed numerically to all
estimators. This distinction is recorded in summary.json.

## Algorithm 1: Nyström Gaussian KRR

Fit a smooth 4D Gaussian radial kernel ridge regression using Nyström
low-rank features. Reuse the same kernel matrix for three response
channels, choose kernel lengthscale and ridge with subject-based
validation, and differentiate the RBF basis **analytically** to obtain
Hessian estimates. Nyström is an *approximate* KRR implementation.

## Algorithm 2: penalized tensor-product cubic B-splines

Use a tensor product of four univariate B-spline bases defined over the
known T supports. Fit a Tikhonov/second-difference roughness-penalized
least-squares surface. Differentiate the actual spline basis exactly
through scipy.interpolate.BSpline.derivative. Supports degree-3 cubic
basis with adjustable number of basis functions per dimension.

## Algorithm 3: second-order integration-by-parts / weighted Stein

This method *does not fit a regression function*.

Because two CR3 priors are truncated with nonzero boundary densities,
the naive unweighted identity E[Hf]=E[f*(score score'+...)] is generally
**false** without boundary terms. Instead define

    z_j=(T_j-low_j)/(high_j-low_j)
    w(T)=prod_j 16*z_j^2*(1-z_j)^2

whose value AND first partial derivatives vanish at all support faces.
Then integration by parts twice gives

    E[w(T) H_ij g(T)] = E[g(T) psi_ij(T)]
    psi_ij(T) = partial_ij (w(T)*pi(T)) / pi(T).

This is a **weighted mean Hessian**, not the originally requested
unweighted H_prior. For comparability, both sides are divided by E[w].
The experiment reports sample score estimates vs analytic sample and
population **weighted** Hessians, standard errors over independent
subjects, and a quadrature check of the identity using the deterministic
conditional mean. It NEVER compares this method to an unweighted
Hessian while pretending that the targets match.

The score method can have **large variance**, especially when the true
prior-mean Hessian nearly cancels. Data-only values for binary responses
include sampling noise; `n=5000` may not suffice for reliable 4D
second-order score estimation.

## Fair evaluation

- Split subjects into 70% train / 15% validation / 15% test for KRR
  and B-splines. The test subjects do not enter the hyperparameter search.
- Select smoothing by **function prediction** on validation data, not
  by looking at the true Hessian; then separately report Hessian recovery.
- Report held-out local averaged-TT Hessian RMSE, sample/prior mean-TT
  Frobenius error, MAE, max error and cosine similarity.
- Direct Stein has no local Hessian prediction: only **weighted** mean
  estimates, identity check and subject-cluster standard errors.
- Methods are applied to three channels: real binary choice, exact logit,
  and stored post-sigmoid p.
- Unweighted oracle H_TT is defined even on fixed trials with zero
  monetary gains/losses; X derivatives are deliberately **not** estimated.

## Outputs

`outputs/math/cr3_hessian_estimators/` contains:

- `summary.json`: complete metrics, hyperparameters, target definitions,
  quadrature mass errors and fit statistics;
- `all_matrices.npz`: numeric matrices with descriptive keys;
- labeled CSVs per analytic/method/sample/prior response;
- comparison heatmaps unless `--no-plots` is specified.

## Built-in tests

`test_math.py` verifies: real CR3 analytic H_TT against numerical finite
differences including sure outcomes and zero monetary amounts; Gaussian
KRR analytic mixed derivatives on a polynomial synthetic function;
B-spline basis Hessian against a known quadratic/mixed term; numerical
integration of the weighted Stein identity on a polynomial; prior
quadrature mass and direct score estimator output shapes.

**Scope limitation:** The results are a controlled benchmark of the
stimulus-averaged 4D Hessian. The model-free estimators have statistical
bias and finite-sample variance, and the smoothness assumptions cannot
be avoided merely by supplying more discrete stimulus points.
