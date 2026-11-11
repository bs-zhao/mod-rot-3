# Wave V: all-trial U-RBF times X-dictionary functions

`run_product_response.py --batch` completed CR3 as
`waveV_product_kernel_seed401`. It uses the immutable all-subject U-response
bank, RBF lengthscales 1/2 and 512 candidate centers, with a generic X
dictionary of complete degree 2 plus a 64-center length-2 RBF kernel.
All cross-basis coefficients are free; no rank-one behavioral product or
task-specific equation is imposed. This is kernel ridge, with no GP
posterior/uncertainty claim.

An X interpolation map is selected using training U subjects only.
The final all-subject U coefficient matrix is linearly mapped into that
X dictionary. A separate training-subject-only U fit is reconstructed to
evaluate predictions against the original raw held-out W/Y rows. The
validation subjects' Y do not select the X interpolation ridge.

Actual U gradients, UU Hessians and UX mixed derivatives follow from the
two feature maps' derivatives. Every original trial contributes to raw,
locally balanced, scalar-profiled and tangent-X covariance descriptors.
The numerically shared-X cache is used only after equality is verified
against every person's actual X rows. It changes computation, not data.

| U kernel | Raw logit validation NMSE | Raw probability validation NMSE |
| --- | ---: | ---: |
| RBF length 1 | 0.0161565 | 0.0352302 |
| RBF length 2 | 0.0150766 | 0.0344631 |

Raw validation uses 625,000 rows from 1,250 held-out subjects. Final
functions and metrics use all 5,000 subjects and all 2,500,000 original
rows. X interpolation checks gave NMSE .0239951/.0499848.

No adequate full-plane candidate survived CR3, so other datasets were
skipped. The best passing-plane count and all failures are retained in
the frozen summary. This does not establish success for RBF UX sparsity.

97 tests passed after this addition, including standardized X derivatives,
raw-held-out-row accounting, complete kernel/WY pipelines, output-link
geometry and previous exact all-trial covariance checks.

The next approximation check increases X dictionary resolution while
retaining a common rule; the current X interpolation is a measurable
source of error. Full joint product-reference ANOVA is another distinct
descriptor from the previously tried U-only ANOVA.

Primary background: [Rasmussen and Williams, covariance functions](https://gaussianprocess.org/gpml/chapters/RW4.pdf).
The two-stage dictionary fit and sparsity diagnostics here are custom;
the cited source does not assert their axis-recovery performance.
