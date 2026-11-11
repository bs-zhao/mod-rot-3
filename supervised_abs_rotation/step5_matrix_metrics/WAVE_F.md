# Wave F: exact functional ANOVA under local reference distributions

Run: `run_anova.py --batch`, protocol `waveF_anova_seed386`.
This completed on CR3 with zero full-plane candidates. It reuses verified
Wave C functions fitted to every subject and every recorded stimulus.

Sixteen observed subject coordinates are chosen without Y. Around each
anchor, define an isotropic Gaussian product reference with standard
deviation 0.2 or 0.5 in common standardized U units. The score averages
**within-anchor** component variances. It is not the ANOVA decomposition
of the joint mixture distribution, nor of the empirical dependent U law.

For polynomial approximators, compute coefficients in the normalized
Hermite basis analytically, including exact coordinate transformations.
Group squared coefficients by the subset of coordinates they involve.
For RBF approximators, factor one-dimensional Gaussian kernel moments,
center each factor, and evaluate every subset's component variance using
the output coefficient Gram matrix. All recorded response columns enter
that Gram matrix, normalized by training-subject response variances.

Both methods compute the same reference-defined function decomposition
without asserting a behavioral equation. Variances sum to the local
function variance; the total is a rotation-invariant control. Numerical
sum errors, negative energy, endpoint errors and reference support coverage
are retained. A Gaussian reference probe must have its 16-neighbor radius
within 1.5 times its anchor radius; coverage must be at least 90%.

The reported metrics include tails/entropy of all components, interaction
components and pair components, plus interaction fractions. Following the
user's later clarification, entropy and strength fractions are auxiliary;
Wave I re-scores the frozen component energies with effective support.

The centered-kernel idea is related to
[Durrande et al.](https://arxiv.org/abs/1106.3571). Here the reference
distribution and averaging convention are explicit. No classical
independent-input Sobol interpretation is claimed for the empirical data.

CR3 result: no adequate full-plane candidate. The best passing-plane count
was 2/6 (rbf1 logit, scale 0.2, interaction/pair tail-4; rank 11/37).
Reference coverage at scale 0.2 was 0.998535. Kernel control roundoff is
larger than polynomial roundoff and is explicitly screened.
