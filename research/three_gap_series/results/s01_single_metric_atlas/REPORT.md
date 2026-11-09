# Single Hessian metric atlas

Channel: logit; seed: 20261128; 314 canonical single definitions; 8 exact aliases omitted. Completed models: C1, C2, C3, CR3.

**Current policy: single metrics only. Combinations and weight tuning are paused.**

These are falsification screens, not completed Gap 1 tests. Plane survivors require full rotation optimization. Flat means at least one unresolved plane at the fixed numerical sensitivity. A random rotation beating identity is recorded separately and does not by itself reject the allowed neighborhood.

| Model | Plane counterexample | Flat unresolved plane | Needs joint optimization |
|---|---:|---:|---:|
| C1 | 2 | 216 | 96 |
| C2 | 74 | 191 | 49 |
| C3 | 110 | 156 | 48 |
| CR3 | 303 | 4 | 7 |

## All-four survivors requiring joint optimization

- `hs_ux_rms_soft003`
- `hs_full_rms_soft003`

## Family breakdown

Each cell is counterexample / flat / needs optimization; counts include predeclared statistic and threshold variants.

| Family | C1 | C2 | C3 | CR3 |
|---|---:|---:|---:|---:|
| curvature_support | 0 / 12 / 0 | 0 / 12 / 0 | 0 / 12 / 0 | 10 / 2 / 0 |
| full | 1 / 49 / 38 | 37 / 37 / 14 | 55 / 19 / 14 | 85 / 0 / 3 |
| jd | 0 / 0 / 8 | 0 / 0 / 8 | 0 / 0 / 8 | 8 / 0 / 0 |
| profile | 0 / 0 / 12 | 0 / 0 / 12 | 0 / 0 / 12 | 12 / 0 / 0 |
| third | 0 / 9 / 0 | 0 / 9 / 0 | 0 / 9 / 0 | 8 / 0 / 1 |
| uu | 0 / 79 / 0 | 0 / 79 / 0 | 0 / 79 / 0 | 77 / 2 / 0 |
| uu_fraction | 0 / 6 / 0 | 0 / 6 / 0 | 0 / 6 / 0 | 6 / 0 / 0 |
| uu_graph | 0 / 12 / 0 | 0 / 12 / 0 | 0 / 12 / 0 | 11 / 0 / 1 |
| ux | 1 / 49 / 38 | 37 / 36 / 15 | 55 / 19 / 14 | 86 / 0 / 2 |

## Interpretation limits

- For C1/C2/C3 logit, pure UU and UUU fields are analytically zero. Flatness here is structural, not a WY estimator failure.
- Signed means can cancel; local absolute values, RMS, SD and quantiles answer different questions. Compare their plane curves, not raw loss scales.
- Top-k row and column tails measure different interaction graphs. Every dimension uses the same k.
- The full family applies one penalty to [H_UU|H_UX] at common differentiable W points. It has no separately tuned component weights.
- Pure total-energy controls are retained to expose invariance or coordinate standardization effects; they are not automatically promoted.
- No decoder TEST result is used for this single-metric selection.

Full metric-by-model cells: [single_metrics.csv](single_metrics.csv). Per-model plane curves and random counter-directions accompany each atlas.
