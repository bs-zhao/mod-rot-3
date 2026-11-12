# Conditional X feature witnesses followed by GP complexity

AT runs CR3 first, then C1/C2/C3 only while an identical primary criterion
survives. It is an exploratory new protocol, not an accepted result.

The previous X omission regression used Y to tune prediction of a response
from the other X columns. AT instead constructs a numerical conditional
feature witness using W alone:

    h_j(X) = phi_j(Xj) - estimated E[phi_j(Xj) | X-minus-j]
    Psi_j(X) = h_j(X) tensor (1, psi_minus_j(X-minus-j)).

RBF feature maps use per-column standardization and median positive pairwise
squared-distance bandwidths. Conditional means use nested five-fold kernel
ridge regression with a free intercept. Ridge is selected only by heldout
X-feature prediction from .0001/.001/.01/.1/1. No Y, case angles or baseline
preference enters the construction or tuning of these operators. A context
tensor product retains effects such as Xj*Xk that marginal feature covariance
could miss. The centered witness Gram is positive semidefinite, with unit
trace per X column. Factors and numerical audits are saved.

The actual double-centered response table is multiplied by each witness
factor to obtain a U-dependent conditional feature covariance. This is an
empirical structural diagnostic inspired by conditional kernel covariance,
not a calibrated KCI hypothesis test or a causal attribution. Finite-sample
conditional-mean error can leave leakage, which is not assumed to vanish.

The same direct-matrix GP implementation and fidelity gates as AR follow:
128 centers, three starts, 30 training and 60 all-5000-subject refit steps,
1e-6 output-mode squared-energy budget, <=.01 heldout and refit relative
MSE for every X task/case. Primary scores are scale-normalized small-tail
and effective-support summaries of inverse-lengthscale relevance. Budget
one, budget-curve area and 30% support counts remain controls.

Run:

```cmd
conda activate py11
cd /d D:\Projects\mod-rot-3
python supervised_abs_rotation\step5_matrix_metrics\run_matrix_gp_ard.py --batch --X-contrast conditional_witness --run-name waveAT_conditional_X_witness_GP_seed416
```

Four tests check primal/dual regression equivalence, exact context-product
feature algebra, PSD/centering/affine-X-unit invariance, and a complete
actual-matrix-only pipeline. The full suite passed 171 tests before launch.

Primary background: Zhang et al. (2011),
[Kernel-based Conditional Independence Test](https://proceedings.mlr.press/r9/zhang11b.html);
Rasmussen and Williams,
[GPML chapter 5](https://gaussianprocess.org/gpml/chapters/RW5.pdf).
