# Differential program rotation: objective research suite

This independent suite fits only frozen decoder fields at NP TRAIN anchors.
No true theta or mechanism labels enter the losses, basis selection, stopping
or candidate choice. The primary of each path is frozen in its config.
All results on existing data remain exploratory.

## 1. Full local differential module code

Let eta=logit1-logit0 be the canonical Bernoulli natural parameter, q be
TRAIN PCA-whitened latent coordinates, and s=(X-mean_X)/sd_X using FIT only.
Compute exact frozen decoder tensors T[r,s]=d^(r+s)eta/dq^r dX_std^s for
(r,s)=(1,0),(2,0),(3,0),(1,1),(1,2),(2,1).
No response-profile surrogate is fitted. Pure powers and mixed products are
both present, with symmetric tensor multiplicities retaining Frobenius energy.

Pack each symmetric tensor into an orthonormal symmetric-tensor basis.
For each latent/stimulus multi-index module, let g be its RMS over anchors,
weighted by h^(r+s)/(r!s!), h=0.5. Divide all g by the joint Frobenius norm,
which is invariant to orthogonal latent rotation. With M modules and fixed
tau=0.02/sqrt(M), optimize

    L_code(A) = sum_module log(1 + g_module(A)/tau).

The predeclared group-L1 ablation minimizes sum g. This is an operational
concave description-length proxy, not a universal symbolic-program code.
A product ab occupies one symmetric Hessian module; rotated squares occupy
two. Penalizing only off-diagonal curvature would have the opposite bias.
Physical raw X coordinates are specified by the dataset; rotating those input
coordinates would change this structural prior. There is no arbitrary
response-profile channel basis.

Run `python rotations/differential_program/run.py --model C1`.

## 2. Axial differential recurrence

At each actual X, form columns [h D_i eta, h² D_i² eta/2!,
h³ D_i³ eta/3!] over subject states. Minimize total residual energy to the
best rank-one approximation, divided by the rotation-invariant energy of
the full first/second/third tensors. This asks how many state modes are needed
to generate the derivative sequence, not whether cross-axis interaction is zero.

For a*b, both true axes have no second derivative. A 45-degree mixture
produces a varying first derivative and constant second derivative, adding a
state mode. For a*x^b, true axes also have rank-one recurrences; mixed axes
generally need two. This does NOT identify linear-in-parameter models: every
axis then has only a first derivative, in every rotation. It is therefore an
interaction-structure hypothesis and diagnostic, not a complete common method.

This candidate runs with the local path above. No data-dependent mixing of it
with module code is introduced.

## 3. Integrated stimulus functional module code

When local Taylor approximation in X is poor, do not increase its objective
weight. Evaluate eta and exact latent derivative tensors on 416 actual FIT X
anchors. Reserve 96 outer X points. In 320 fitting X points, use inner held-out
X to choose polynomial degree 2/3/4 and ridge from a fixed grid by mean relative
MSE of eta and the complete latent gradient. All polynomial monomials of each
degree are included and normalized from FIT X; no model-specific mechanisms
or stimulus flags are used. This is a finite dictionary approximation.

Project only in X. There is no fitted latent polynomial. Latent derivative
coefficient tensors at every subject state are transformed exactly under A.
Apply the same module code (and predeclared L1 ablation) to their functional
coefficients. Full tensor projection residuals have an isotropic Frobenius
escape cost, invariant under A because the fixed X projection and latent
rotation commute. Hence a candidate cannot obtain a smaller cost by being
fitted less accurately. Outer-X value/gradient/curvature errors are reported;
poor fidelity still invalidates a structural interpretation.

Run `python rotations/differential_program/functional_run.py --model C1`.
This path has its own fixed `functional_config.json`. Its primary is
functional_code; the secondary functional_l1 is not selected by theta.

## Function-error calibration, not theta-based weight tuning

`calibrated_run.py` first selects the largest latent and stimulus perturbation
radii on a fixed descending grid that achieve at most 1% relative Taylor effect
error on both FIT calibration halves. It then uses hq^r hx^s/(r!s!) in the same
module code. The same grid/cap/selection rule applies to every model. The
orthogonal matrix remains constrained; this does not optimize latent scaling.
The radius is frozen before A is fitted. A smaller X radius is a numerical
validity correction, not evidence that parameter axes have become identified.

## A further structural objective: output equivariance

`symmetries.py` formalizes two functional null-space objectives using eta and
G on actual common stimuli. Translation uses G_i-mean_X G_i. Gain uses
G_i-beta_i eta where beta_i=<eta,G_i>/<eta,eta>, normalized by RMS eta.
Minimize each residual relative to that direction's full derivative energy,
using a generalized Rayleigh quotient. This prevents a weak direction from
being mistaken for a symmetry merely because all its effects are small.
The direction minimizing each quotient is a candidate globally additive or
multiplicative effect. No parameter identity or stimulus flags are used.

Accept only a unique, active direction with residual eigenvalue ratio <=0.2,
eigengap exceeding twice the split-half operator uncertainty, and nontrivial
gradient energy. A rank-deficient gain field alone cannot identify a unique
axis: e.g. a*b*x has a whole ambiguous scale subspace and is rejected. These
function-supported axes constrain a subsequent module-code search in
`equivariant_run.py`. If no property is supported, the same procedure falls
back to unconstrained module search. Accepted directions are fixed first;
only their orthogonal complement is searched. The primary is
`equivariant_code`, with the predeclared `equivariant_l1` ablation. Numerical
thresholds and order of testing are identical across all benchmarks.

The generalized quotient uses full derivative energy, not just a smallest
ordinary covariance eigenvalue. A weakly active direction is not sufficient
evidence for a functional symmetry. Tests include the nonidentifiable product
counterexample and exact additive/multiplicative toy properties.

## Mathematical gates and numerical contract

`python -m rotations.differential_program.toys` checks angle curves before
benchmark evaluation. Both module variants prefer true axes for a*x^b,
a*b*x, and a+b*x. Recurrence is correctly flat on the last toy. Tests verify
symmetric-power rotation, derivative direction and nested AD.

Column convention q=W(z-mu), u=Aq; rows U=Q A^T. W includes whitening,
A alone is the orthogonal rotation. C1/C2/CR3 use 4 PCs, C3 6 of 8 native
latent dimensions. Inverse decoder z=mu+W^+ A^T u+r retains the subject's
unselected residual. Full prediction-distribution equivalence is checked.

Save every A and mapping before theta evaluation. Post-fit metrics include
TRAIN-defined matching applied to TEST, TEST-optimal matching as a separate
diagnostic, complete signed correlation, every parameter, off-target loading,
all initializations and a TRAIN supervised reference. Angle curves through
the supervised reference are diagnostic only.

## Relation to previous work

The [Hessian Penalty](https://arxiv.org/abs/2008.10599) encourages a diagonal
Hessian. That prior is not adopted here because genuine products must remain
available. [Equation learning](https://proceedings.mlr.press/v80/sahoo18a.html)
motivates sparse functional descriptions. The objectives here are differential
module proxies, not the EQL architecture or proof of symbolic identifiability.
