# Wave L: direct joint U/X coefficient support

Run `run_joint_projection.py --batch --device cuda`.
The fixed model sequence is CR3, C1, C2, C3. A candidate must retain the
identical degree, channel and score in every dataset; the intersection is
updated after each model and an empty intersection stops the batch.

Fit generic complete polynomial projections of degrees 2/3/4 on every
original W/Y row. These are explicitly low-order matrix statistics, not
asserted behavioral equations. Each U uses one common scalar scale; each
X coordinate is standardized from training rows. Multinomial feature
weights make the kernel isotropic and preserve exact U-rotation equivariance.
Select ridge by subject-held-out normalized error, then refit all subjects.

A recurrence generates the exact same weighted monomials without repeated
exponentiation. CUDA accumulates float64 sufficient statistics; CPU tests
and a real CUDA test verify equivalence. It does not train or read a neural
network. A NumPy backend remains available. Chunking affects memory only.

Transform coefficients by homogeneous U-degree and fixed X multiindex.
The algebraic transformation uses generic basis probe points only to
compute a coordinate-change matrix; no new response observations are
synthesized. Its orthogonality and prediction equivalence are tested.

Report Wave H's scale-free effective support for all U-containing terms,
UX terms, cross-U terms, U-axis-support groups, joint U/X support groups,
and X-column coefficient profiles. X-only nuisance coefficients do not
select U axes. There are no model-specific function flags, signs or mixture
weights. Coefficient Frobenius energy is an invariant control.

This is support in a declared polynomial dictionary, not an empirical
Sobol decomposition. Correlated/aliased dictionary columns can split
coefficients; prediction-compression experiments are therefore a distinct
next check. No coefficient-support pass alone establishes function recovery.

The CR3 run completed with no full-plane candidate. Its best setting
reached 5/6, rank 3/37; all failures and complete scores remain preserved.
