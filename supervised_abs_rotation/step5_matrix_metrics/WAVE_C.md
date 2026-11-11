# Wave C: fixed-X direction consistency and finite-scale effects

Frozen protocol: `waveC_finite_seed383`, CR3 first. This experiment tests
different properties from additive decomposition. It does not assume that
the unknown response is additive, monotone, polynomial, or a GP draw.
Polynomial and RBF functions are explicit numerical approximators whose
subject-held-out error is reported, not asserted true equations.

## Response bank

If numeric X values repeat in exactly the same order for every subject,
the observed common Y is rearranged to subject-by-stimulus response banks.
Every subject and stimulus is retained for fitting. No equation or decoder
is used. CR3 satisfies this data-driven condition.

For an unshared X design, the generic all-trial profile construction from
Wave B is evaluated at 128 pooled observed X probes. Its separate trial
validation error must pass the same 0.10 screen. Common support of those
X probes across every subject is not asserted; this remains a limitation
of that branch and must be checked if CR3 produces a candidate.

## Generic U-to-response fits

Fit all response columns jointly using isotropic polynomial kernels of
degrees 3/5 and Nystrom RBF kernels with lengthscales 1/2 and 512 training
subject centers. Use a 75/25 subject split to select ridge from the Wave A
grid and refit on all 5,000 subjects. Y prediction NMSE is averaged over
stimuli within each channel, with per-stimulus variance normalization.
All columns enter fitting; a fixed random 64-stimulus subset is used for
effect scoring and explicitly saved. This subset is not selected by score.

The same fitted function is transformed to every anonymous case using only
Procrustes maps inferred from actual W. Coefficients, fields and numerical
settings are saved. A prediction NMSE screen does not certify derivatives.

## Conditional direction consistency

For effect values e over subjects at one stimulus and axis, cancellation
is `(sum(abs(e))-abs(sum(e)))/sum(abs(e))`. Numerators and denominators
are pooled over stimulus-axis groups. Signs can differ freely between
stimuli and U axes. This differs from the earlier single global sign.

To allow signs also to depend on other U coordinates, cluster those
coordinates into 8 or 32 Euclidean groups per axis, using deterministic
response-independent clustering. Calculate cancellation within each
context/stimulus/axis group. This is an empirical local diagnostic, not
a proof of global or conditional monotonicity. Contexts remain finite;
larger groups can conceal or invent apparent sign reversals.

Apply the same score to partial gradients and central finite differences
at standardized U steps 0.2 and 0.5. No coordinate is rescaled separately
after rotation. Signs and coordinate permutations preserve the definitions.

## Support and additional structure

For each anchor and every shifted point, calculate its 16-neighbor radius
in observed U. The shifted radius must be at most 1.5 times the anchor's
radius. Use the intersection of retained anchors over ALL cases, axes,
signs and steps, so comparisons do not select different people per case.
At least 50% of subjects must remain for an adequate candidate. This is a
geometric coverage diagnostic, not a density or causal-support guarantee.
Every subject still contributes to fitting.

Additional fixed metrics include pointwise axis tails, pooled UU Hessian
edge/all-entry tails, directional curvature L1 normalized by the invariant
Hessian Frobenius energy, and U-effect tails of nearest observed-stimulus
contrasts. The last measures a finite UX response contrast; it is not
claimed to equal a partial derivative with respect to a particular X axis.
Gradient/Hessian Frobenius energies are invariant controls, not candidates.

Every candidate uses lower-is-simpler in all datasets and the original
strict baseline-rank/all-plane endpoint criterion. Repeated experiments
on the same four datasets remain exploratory. CR3 failure stops the batch.

```cmd
python supervised_abs_rotation/step5_matrix_metrics/run_finite.py --batch
```

The support concern is motivated by
[Apley & Zhu](https://arxiv.org/abs/1612.08468); this implementation is not
their ALE algorithm. Conditional sign cancellation is a custom diagnostic.
Related shape-constrained regression work, such as
[Chen & Samworth](https://arxiv.org/abs/1404.2957), requires explicit shape
assumptions; no such assumption or theorem is claimed for these data.
