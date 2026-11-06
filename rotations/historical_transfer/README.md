# Historical regression and DDM rotation transfer

This folder is **one complete independent rotation project**. It imports no other
rotation suite and does not retrain or modify the frozen NP. Its own entrypoint
loads frozen full-observed-trial Z, observed X/Y and the trained decoder, fits all
historical attempts, persists every transform, and only then reads simulation
parameters for evaluation.

The completed four-model study, including negative results and all comparison
tables, is in [RESULTS.md](RESULTS.md). Compact numerical evidence is in `results/`.

```powershell
python rotations/historical_transfer/run.py
python rotations/historical_transfer/run.py --models C1 C2 C3 CR3 --representations pca_whitened
python -m pytest rotations/historical_transfer/test_methods.py -q
```

Default: C1/C2/C3/CR3, repeat 1, condition
`beta0p0005_lambda0_ctx0p25to0p8_L8`. No cross-method or cross-model winner is
selected. Every method is reported, including failures and exact aliases.
`--smoke` uses a separate fingerprint and reduced budgets for wiring checks;
smoke results are not research results. `--tag NAME` starts a separate preserved
run. A completed identical fingerprint can be reevaluated without refitting;
incomplete output is preserved and cannot silently be overwritten.

## Historical coverage

Regression sources: user-supplied `rot1_config.zip`, following the original
[regression project](https://github.com/bs-zhao/moderational-learning-dm/tree/31864e32386a93b73e224810441803c2ada2f7f1/simulation_studies).
The archive is a later M1–M4 rotation study and is not identical to the two old
simulation folders. Extracted reference function bodies are retained locally in
`*_reference.py`; archive source SHA256 hashes are in `source_manifest.json`.

Each regression attempt is run on both the binary mean p and the log-odds eta.
The latter is an explicit link-scale control, not a claim that p and eta define
the same functional decomposition.

| Original attempt | Local method suffix |
|---|---|
| rotation/s1 point-mean Fisher-Geomin | gradient_geomin |
| rotation/s2 linear XZ | linear_xz_geomin |
| rotation/s3 level + linear XZ | level_linear_xz_geomin |
| rotation/s4 nonlinear spline/pairwise XZ + level | nonlinear_basis_xz_geomin |
| rotation/s5 quantile-bin contrasts + level | nonparametric_contrast_geomin |
| rotation/s6 equal-weight joint library | joint_signature_geomin |
| auto A2 level + rich functional basis | basis_geomin |
| auto A2 basis + gradient block | basis_joint_geomin |
| auto A2b level anchor + entropy | anchored_entropy |
| auto A2b entropy + weak coverage | anchored_entropy_coverage |
| auto A2c detected shared-X groups + whole-function curvature | functional_shape |

The original `a1_auto_subspace.py` is a preprocessing stage, not another rotation.
TRAIN-only PCA90, permutation parallel analysis and eigengap diagnostics are
saved, while controlled K is fixed at 4/4/6/4 for all methods within each model.
This avoids confounding the objective comparison with variable latent dimension.

DDM sources:
[linear attempts](https://github.com/Cognition-Decision-Modeling-Lab/latent_choice_rt-v1/tree/2f5b1e921084c687ef01b3c0486b81920c429507/experiment/sim1_ddm/linear_rotation),
[continuous Fisher](https://github.com/Cognition-Decision-Modeling-Lab/latent_choice_rt-v1/tree/2f5b1e921084c687ef01b3c0486b81920c429507/experiment/sim1_ddm/continuous_rotation),
[observed-X Fisher](https://github.com/Cognition-Decision-Modeling-Lab/latent_choice_rt-v1/tree/2f5b1e921084c687ef01b3c0486b81920c429507/experiment/sim1_ddm/observed_x_rotation).

| Original attempt | Local method |
|---|---|
| choice/RT fingerprint Jacobian + SVD + Kaiser Varimax | ddm__fingerprint_svd_varimax |
| general invertible JVolmax | ddm__jvolmax_oblique |
| orthogonal constraint control, added for current project contract | ddm__jvolmax_orthogonal |
| fixed probe grid Fisher | ddm__fixed_common_x_fisher_{geomin,quartimin} |
| integrated design, channelwise Fisher | ddm__integrated_x_fisher_{geomin,quartimin} |
| integrated design, joint Fisher | ddm__integrated_x_joint_fisher_{geomin,quartimin} |
| every actual FIT trial X Fisher | ddm__observed_x_fisher_{geomin,quartimin} |

The historical `s2_diagnose_linear_rotation.py` is diagnostic, not a separately
fitted rotation. SVD spectrum, feature scales, finite-difference/autodiff
agreement, correlations, off-target correlations, inverse and prediction audits
are retained here. Data generation, NP training and oracle parameter fits are
not rotation attempts and are not rerun.

## Required adaptations, recorded rather than hidden

1. **No RT channel exists** in these four binary benchmarks. Distributional
   fingerprints retain choice log-odds only. No synthetic RT channel is invented.
   Binary choice has a single Fisher component, so joint and channelwise Fisher
   are mathematically identical. The joint labels are persisted aliases, not
   independent supporting experiments.
2. **Multivariate X replaces scalar coherence.** Fixed probes are 64 shared
   observed FIT X rows. For C1/C2/C3 the integration control uses order-5 tensor
   Gauss–Legendre quadrature over independent uniform coordinates on observed FIT
   bounds (125/625/625 nodes). This product measure is an explicit assumption.
   For CR3 the shared observed lottery bank is discrete and constrained;
   integration is the exact uniform sum over its unique joint rows. A fictitious
   continuous rectangle is never used for CR3. Details/nodes/weights are saved.
   With the complete common CR3 bank and equal trial counts, integrated-X and
   observed-X also define the same empirical objective; separate optimizer seeds
   may expose local-optimum variation, not new information or independent evidence.
3. **Two coordinate systems are fully compared.** `pca_whitened` is the current
   project's convention; `raw_top_standardized` is the historical DDM convention.
   Selection, PCA and scale estimates use TRAIN only. All gradients use original
   subject Z with the complete residual preserved. K is fixed by configuration.
   The fingerprint SVD/Varimax is therefore fitted inside each controlled K-space;
   the old script's additional full-native-Z to top-K SVD truncation is not a
   third subspace-selection experiment here. This isolates objectives within a
   fixed coordinate family rather than claiming a byte-for-byte pipeline rerun.
4. **JVolmax is originally oblique.** Its result is labeled invertible-oblique;
   orthogonality is not asserted. The separately optimized orthogonal version is
   an adaptation, not a replacement. Both use the exact inverse.
5. **Finite optimization budget:** all orthogonal fits use 8 deterministic
   starts, maximum 500 L-BFGS iterations, with analytic derivatives through matrix
   exponentials. Historical code used differing budgets and finite differences
   or Adam. Criteria, normalization and block weighting are retained. JVolmax
   retains multi-start Adam, degree-16 homogeneous mean, 3000 steps, determinant
   scale normalization, patience and a numerical condition bound. Convergence
   records and all restart matrices are saved.
6. Regression geometry uses 128 TRAIN subjects × 40 actual trials/person, with
   1500 rows for the raw-gradient criterion and 500 in the joint library. DDM
   observed-X Fisher uses **every valid trial** from the same 128 subjects, with
   equal total weight per person. All method settings are identical across models.
7. Mean-gradient geometry is only a point-mean analogue of Fisher. Binary Fisher
   is computed analytically as p(1-p) ggᵀ for g=∇eta. After per-cell trace
   normalization its rotated diagonal equals squared projections of unit g;
   this exact rank-one simplification is tested against full matrices.

There are 34 saved labels per representation: 22 regression attempts, 9 distinct
DDM attempts, 2 exact joint-Fisher aliases and identity. Across both coordinate
systems and four models, 272 labels are reported; aliases are never counted as
independent evidence.

## Transform and evaluation contract

Column convention: u=Aq. Row-stored U=Q Aᵀ. Original z is reconstructed with

```
q = inverse(A) u
z = mu + B q + residual
B = pinv(W), q = W (z-mu)
```

Every method persists A, its inverse, W/B/mu, U, split IDs and FIT subject IDs
before parameter access. Per-method numerical checks cover inverse consistency,
latent roundtrip, choice log-odds/probability equivalence and orthogonality where
applicable. Every requested model/representation is fitted before any evaluation
is entered. Ground-truth parameter arrays are accessed only by `evaluation.py`.

Primary statistic: Hungarian one-to-one matching on **TRAIN** squared Pearson
correlations, followed by **TEST** Pearson r² under that fixed matching. Full
signed TRAIN/TEST matrices, per-parameter scores, mean/minimum target r²,
off-target mean r² and maximum absolute r are saved. TEST-optimal matching is
explicitly diagnostic. No method is selected using TEST results. These statistics
must not be conflated with the old projects' evaluation-set matching or CV R².

## Files and outputs

- `data.py`, `decoder.py`: local frozen-data/decoder loading.
- `fields.py`: representations, observed/common/integration fields and derivatives.
- `*_reference.py`: original regression function bodies, adapted imports only.
- `optimization.py`: differentiable objectives and optimizers.
- `methods.py`: complete historical attempt inventory.
- `run.py`: fitting, immutable fingerprints, persistence and equivalence audits.
- `evaluation.py`: post-fit truth diagnostics and signed heatmaps.
- `test_methods.py`: Fisher identity, entropy parity, volume scale invariance,
  oblique inverse, analytic-gradient check and shared-group curvature tests.

Generated artifacts stay under ignored
`outputs/choice/<MODEL>/rotations/historical_transfer/<condition>/repeatNN/`.
Each immutable run contains protocol/source/input hashes, design nodes/fields,
A1 diagnostics, and one subfolder per method with mapping, starts, fit/evaluation.
Combined comparison CSV and run paths are under
`outputs/choice/historical_transfer/summary_<fingerprint>/`.
