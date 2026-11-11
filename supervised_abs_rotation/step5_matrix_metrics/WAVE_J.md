# Wave J: spectral support of functional separation modes

Protocol `waveJ_separation_seed389`, run `run_separation.py --batch`.
The question is whether each U axis enters the response through only a
few separable function terms. A product of univariate functions can have
interactions everywhere while still requiring a single separable term.
This is a broader composition descriptor than sparse interaction edges.

For each axis, center its functional dependence under an explicitly
declared local Gaussian reference. View the resulting response as an
operator from that axis to all remaining U coordinates and all stimulus
outputs. The operator's squared singular values are its separation
spectrum. Overall output scale cancels from all support metrics.

Use the exact polynomial Hermite coefficient Gram matrix to construct
each axis-mode covariance. For RBF functions, combine analytic moments
on other axes with 12-node Gaussian quadrature on the selected axis.
Compare quadrature total energy to the analytic total. Check negative
eigenvalues and remove energy below a fixed 1e-8 of mode trace as a
numerical spectral floor. This floor is identical across every case,
model and estimator; sensitivity checks are needed for any survivor.

Apply Wave H's effective support, soft/fractional support counts and
tail-2 L2 to each spectrum. Average equally over eight Y-independent
observed anchors, then summarize the mean and worst axis. This avoids
giving a large-effect axis more weight merely because it is stronger.
Use scales 0.2/0.5, the common 90% reference-coverage requirement and
the same prediction/strict endpoint screens as previous waves.

Every underlying function was fitted on all 5,000 subjects and every
response column. Only the integration-anchor count is eight; it does not
reduce fitting data. The function loader checks source score integrity,
reconstructs kernel coefficients and verifies their gradients against
saved fields before use. No original decoder weights or saved R are read.

This experiment is motivated by functional low-rank representations, e.g.
[Gorodetsky, Karaman & Marzouk](https://arxiv.org/abs/1510.09088).
It measures mode spectra of already fitted functions; it does not claim
to implement their functional tensor-train learning algorithm or its
guarantees. No low-rank behavioral equation is assumed true.

A CR3 survivor must agree across polynomial and RBF families before the
other datasets launch. A surviving four-model metric would still need
anchor, spectral-floor and estimator-resolution checks, resampling and
blind full-dimensional rotation optimization.

## CR3 result

The cross-family gate failed. There was one adequate single-estimator
pass: poly3 logit `s0.2_axis_worst_effective_support_p0p5`, baseline rank
1/37 and 6/6 planes. The same score under poly5 was rank 8/37 with 0/6;
rbf1 was rank 6/37 with 1/6; rbf2 was rank 7/37 with 0/6. It is therefore
an approximation-sensitive lead, not an accepted result. The probability
channel did not reproduce the full pass. Other models were not launched.

Scale 0.2 had reference coverage 0.9921875; scale 0.5 had only 0.859375
and was excluded by the frozen 90% adequacy requirement. The best other
adequate setting was rbf2 logit mean fractional support at 10% L2 error:
rank 4/37, 5/6 planes. No failed plane or estimator was discarded.
