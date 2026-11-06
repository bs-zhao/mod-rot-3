# Stimulus-contrast covariance rotation

**Research status: rejected as a general parameter recovery criterion.** Exact
linear and power-function counterexamples prefer mixed coordinates. The method
is retained for reproducibility. See the dated research log under `research/`.

Exploratory frozen-decoder research, primary fixed before new theta evaluation:
`contrast_logit`. No mechanism flags, parameter labels, or true theta enter fit.
The same objective/configuration applies to every benchmark.

For G[n,i,x] = d(logit1-logit0)/dq_i, form C[x]=mean_n G[n,:,x]G[n,:,x]^T
and M[x]=C[x]-mean_x C[x]. Minimize the total squared off-diagonal energy
of A M[x] A^T divided by sum_x ||M[x]||_F² over orthogonal A.
The common covariance can reflect global decoder sensitivity rather than
stimulus-dependent functional roles. Removing it tests that explanation.
All channels here are raw shared stimuli, not polynomial stimulus features.
The denominator is rotation invariant; off-diagonal energy is not. Constant
or isotropic context covariance is unidentifiable. A bilinear interaction
with isotropic subject states is a counterexample, not a recovery theorem.

This suite has its own truth-free verified NP/PCA loader and local frozen
decoder AD utilities, copied with attribution from the historical teacher.
It never imports or launches another rotation suite during fitting.
Its optimization and reporting are local. Historical diagnostic audit scripts
can compare old results. Existing cached derivative arrays are reused after matching
latent/checkpoint hashes, PCA tangent vectors, FIT subject IDs and config.

Column convention: q=W(z-mu), u=Aq; row convention U=Q A^T.
W contains PCA whitening. A is the k-by-k orthogonal matrix; A W is the
full affine linear map, not an orthogonal raw-latent transform.
Let B=W^+=P diag(sqrt(eigenvalues)), r=z-mu-Bq. Decoder inverse is
z=mu+B A^T u+r. Residual r is retained, so rotation preserves the full
categorical distribution. The zero-residual truncated prediction is audited
separately and never attributed to rotation.

Run from repository root:

```
conda run -n py11 python rotations/stimulus_contrast/run.py --model C1
conda run -n py11 python rotations/stimulus_contrast/run.py --model CR3
```

All fits/rotations are saved before theta is loaded for post-fit diagnostics.
Identity, pooled covariance JD and probability contrast are fixed ablations,
never selected using theta. Reference TRAIN matching and TEST-optimal
matching are both reported. Supervised reference is a separate diagnostic,
fitted only on TRAIN theta. Observed datasets are exploratory; additional
anchor seeds do not establish independent NP replication.
