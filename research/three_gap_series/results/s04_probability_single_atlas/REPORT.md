# Single Hessian metric atlas

Channel: probability; seed: 20261128; 314 canonical single definitions; 8 exact aliases omitted. Completed models: C1, C2, C3, CR3.

**Current policy: single metrics only. Combinations and weight tuning are paused.**

These are falsification screens, not completed Gap 1 tests. Plane survivors require full rotation optimization. Flat means at least one unresolved plane at the fixed numerical sensitivity. A random rotation beating identity is recorded separately and does not by itself reject the allowed neighborhood.

| Model | Plane counterexample | Flat unresolved plane | Needs joint optimization |
|---|---:|---:|---:|
| C1 | 185 | 10 | 119 |
| C2 | 245 | 8 | 61 |
| C3 | 255 | 10 | 49 |
| CR3 | 305 | 4 | 5 |

## All-four survivors requiring joint optimization

No all-four survivor.

## Family breakdown

Each cell is counterexample / flat / needs optimization; counts include predeclared statistic and threshold variants.

| Family | C1 | C2 | C3 | CR3 |
|---|---:|---:|---:|---:|
| curvature_support | 0 / 4 / 8 | 1 / 3 / 8 | 2 / 2 / 8 | 9 / 3 / 0 |
| full | 65 / 0 / 23 | 78 / 2 / 8 | 86 / 2 / 0 | 88 / 0 / 0 |
| jd | 0 / 0 / 8 | 2 / 0 / 6 | 1 / 0 / 7 | 8 / 0 / 0 |
| profile | 0 / 0 / 12 | 0 / 0 / 12 | 0 / 0 / 12 | 12 / 0 / 0 |
| third | 8 / 0 / 1 | 9 / 0 / 0 | 9 / 0 / 0 | 7 / 0 / 2 |
| uu | 60 / 3 / 16 | 66 / 2 / 11 | 64 / 4 / 11 | 78 / 1 / 0 |
| uu_fraction | 4 / 0 / 2 | 6 / 0 / 0 | 6 / 0 / 0 | 6 / 0 / 0 |
| uu_graph | 12 / 0 / 0 | 12 / 0 / 0 | 12 / 0 / 0 | 9 / 0 / 3 |
| ux | 36 / 3 / 49 | 71 / 1 / 16 | 75 / 2 / 11 | 88 / 0 / 0 |

## Interpretation limits

- For C1/C2/C3 logit, pure UU and UUU fields are analytically zero. Flatness here is structural, not a WY estimator failure.
- Signed means can cancel; local absolute values, RMS, SD and quantiles answer different questions. Compare their plane curves, not raw loss scales.
- Top-k row and column tails measure different interaction graphs. Every dimension uses the same k.
- The full family applies one penalty to [H_UU|H_UX] at common differentiable W points. It has no separately tuned component weights.
- Pure total-energy controls are retained to expose invariance or coordinate standardization effects; they are not automatically promoted.
- No decoder TEST result is used for this single-metric selection.

Full metric-by-model cells: [single_metrics.csv](single_metrics.csv). Per-model plane curves and random counter-directions accompany each atlas.
