# A globally integrable output-link correction for UX support

AM is running CR3 first, using the complete degree-four/five numerical
joint-W approximations and all 5000 subjects/all recorded rows. It supplies
no generating equation or output-link formula.

Earlier E/U removed a separate scalar multiple of g*g' at each point.
Those local parameters need not correspond to one function of the output.
AM instead estimates a single scalar function of the fitted response y:

    A = g_U g_X^T
    k(y) = - E[<H_UX,A> | f=y] / E[||A||_F^2 | f=y]
    R = H_UX + k(f) A.

Because log h'(y) is the integral of k(y), this defines one positive
derivative and hence one globally monotone output transformation on the
observed output interval. R equals H_(h o f)/h'(f). It is a data-driven
correction of output coordinates, not a claim that interactions vanish.
The coefficients cannot vary independently with W, X or a rotation case.

There is a useful population chain-rule property. For an increasing
response reparameterization s(f), with a=s' and b=s'',

    H_new = a H + b A,  A_new = a^2 A,
    k_new(s(y)) = k(y)/a - b/a^2,
    R_new = a R.

Pointwise X-column normalization removes this last positive scalar. The
population construction is therefore output-link invariant. Finite-bin
conditional estimates and independently fitted polynomial derivatives are
not claimed to inherit exact finite-sample invariance. The algebra and U
rotation equivariance are tested directly.

Conditional moments use 0 (identity),16,64,256 quantile bins and interpolate
k linearly between bin centers. A fixed 75/25 subject split selects the
resolution by held-out mixed-derivative residual energy; case labels and
preferred angles never enter this choice. That resolution is refitted on
all subjects. Each channel's selected resolution is the primary procedure;
individual resolutions are controls for sensitivity inspection.

The numerical integral records extrema of log h', including zero crossings
of k within interpolation intervals. An unrepresentable derivative dynamic
range fails the numerical guard. Prediction/gradient/mixed-derivative
caches cover all original rows. Corrected UX columns are normalized at
each observed point before per-person covariance. Effective support and
relative-tail counts are then scored with the same signs on every dataset.
Strength is used to estimate the nuisance correction, not as the primary
rotation score.

This is a new chain-rule/conditional-projection construction, not a claimed
implementation of ACE or a proof of rotation recovery. Nonparametric
response transformations provide methodological context (Breiman and
Friedman, 1985):
https://www.tandfonline.com/doi/abs/10.1080/01621459.1985.10478157 .
Derivative accuracy, polynomial/RBF agreement, output-reparameterization
stability, bootstrap margins and blind rotation minimization remain
necessary if a grid criterion survives.

Three tests pass before the real launch: chain-rule and rotation identities,
global response-only recovery/integration, and a full W/Y-only pipeline
checking every cached derivative against an independent evaluator.
