# Wave E: locally profile unknown output-link curvature

Protocol: `waveE_profiled_seed385`, using Wave C's saved all-subject
numerical gradient/Hessian fields. No behavior formula, decoder or theta
is used. Actual W is reread to infer the maps from the source coordinate
system to fresh anonymous cases. The source's frozen score digest and
current derivative-field file hashes are recorded.

## A nuisance that ordinary Hessian sparsity mixes together

For a scalar output transformation with first/second derivatives a/b,
the gradient and Hessian change as `g_new=a*g` and
`H_new=a*H+b*g*g.T`. An unknown monotone output link can therefore add a
dense rank-one curvature term even if the underlying relation is simple.
This chain rule requires no particular link formula.

At each observed subject/stimulus, profile a scalar lambda from
`H-lambda*g*g.T`. Two fixed rules are used:

- L2 projection: choose lambda minimizing squared off-diagonal residuals.
- L1 projection: choose the weighted median of `H_ij/(g_i*g_j)`, with
  weights `abs(g_i*g_j)`, minimizing absolute off-diagonal residuals.

Normalize the entire residual by its Frobenius norm, then measure pooled
and pointwise tail-L2/entropy of all UU entries and of UU edges. Also
report residual off-diagonal fraction and directional curvature L1.
Every metric uses lower-is-simpler for all four datasets.

In exact arithmetic, absent degenerate zero residuals, the profiled unit
residual is unchanged up to sign by a smooth monotone output
reparameterization. Numerical tests explicitly verify this chain-rule
property, signed-permutation invariance, and the additive-function case.

## Limits of this local compression descriptor

There is one nuisance scalar for every subject/stimulus. The scalar is
not required to depend only on Y. Thus this is a flexible *local curvature
compression score*, not proof that an integrable global link exists, and
not a solved unknown-link additive model. Any apparent candidate requires
checking lambda's predictability from Y, derivative resolution, subject
resampling and arbitrary rotations before being accepted.

Both the original estimator's prediction adequacy and strict all-plane
endpoint tests remain in force. The same metric must survive all four
datasets and both numerical approximation families. CR3 failure stops the
batch. Earlier failed scores remain preserved.

```cmd
python supervised_abs_rotation/step5_matrix_metrics/run_profiled.py --batch
```

This is distinct from Wave D: Wave D estimates a global monotone response
calibration using training observations and scores held-out original Y;
Wave E removes an unrestricted local chain-rule nuisance from estimated
derivatives and is correspondingly a weaker structural interpretation.
