# R: first common grid candidate; S: stability failure

`run_local_joint.py --batch` completed as `waveR_local_joint_seed397`.
CR3 was screened first, then C1/C2/C3. The identical projection degree,
response channel and score intersection has exactly one survivor:

**poly4 / logit / UX_columns_subject_median_fractional_count_eps010**.

| Dataset | Baseline rank | Passing planes | Response validation NMSE |
| --- | ---: | ---: | ---: |
| CR3 | 1/37 | 6/6 | 0.00892709 |
| C1 | 1/37 | 6/6 | 0.0000809827 |
| C2 | 1/37 | 6/6 | 0.00146661 |
| C3 | 1/91 | 15/15 | 0.00294670 |

All original W/Y rows and all 5,000 subjects fit each frozen joint
projection. No behavioral equation, original decoder, true theta or saved
R is read. The complete isotropic polynomial dictionary is an explicit
approximation choice. Source banks, score files, code revisions and
actual float64 CUDA backends are checked by `compare_prepared_runs.py`.
The strings `auto` and `cuda` are dispatch requests; the actual recorded
backend is identical and independently checked for every fitted degree.

## Definition of the surviving score

At eight recorded trial positions spaced through each person's original
trial rows, compute the actual mixed derivative H(Ui,Xj) of the fixed
projection in standardized coordinates. Derivatives are obtained by exact
polynomial coefficient lowering; tests match the original derivative code.
U uses one common scalar scale, and X uses training-row standard deviations.

For each subject and X column, average H(Ui,Xj)^2 across those contexts,
then normalize over U axes to sum to one. Sort these energies and compute
the interpolated number of U terms retaining 99% of that column's energy
(10% relative L2 tail). Take the subject median separately for every X
column, then average columns. Lower is simpler. All columns and subjects
are treated by the same rule; no model-specific signs, weights or thresholds
are used. Overall response amplitude cancels.

This directly prioritizes effective sparsity over strength and top-one
dominance. It is not an empirical Sobol index. The source fit uses every
trial; the derivative descriptor initially samples eight trial contexts
per person. Those are distinct coverage claims.

R also preserved every other local UU/UX candidate and its failures.
The number of passing settings narrowed from 16 on CR3, to 16 common
after C1, to one after C2, and one after C3. Near-passes were not counted.

## S: prospectively fixed validation

Run `validate_ux_column.py --batch`. This refuses to launch unless the
fixed lead passed all four original grids. It uses the same polynomial
fit but increases observed contexts to 32 per person. Its covariance
objective reproduces the full derivative-field score exactly; piecewise
analytic gradients are checked by finite differences on the rotation group.

The fixed validation protocol uses 16 random/anonymous-reference starts,
128 Haar random rotations, and 400 paired subject bootstraps. The optimizer
receives only the function field and fixed scoring rule. Its results are
frozen and hashed before baseline/angle evaluation. Axis matching allows
signs and permutations. BFGS convergence flags and precision-loss messages
are retained; a numerical optimizer warning is not recast as a proof of
successful axis recovery or of a global mathematical minimum.

The bootstrap reports a simultaneous minimum interior-angle margin under
the frozen fit. It does not refit the response model and therefore does
not cover function-estimation uncertainty. Higher approximation resolution
and actual optimization recovery must still be assessed.

S completed CR3 and rejected the candidate. Increasing contexts from 8
to 32 changed its baseline rank to 6/37 and passing planes to 2/6.
The paired-bootstrap simultaneous minimum-margin 5th percentile was
-0.0550581; no bootstrap replicate kept every interior margin positive.

The baseline loss was 3.64068044, compared with random-rotation minimum
3.40093147. That explicit counterexample is sufficient to reject global
axis recovery. The best blind BFGS result was 3.33153902, but all 16 BFGS
starts reported precision loss on the nonsmooth median/count objective;
they are not claimed as converged global optima. The best score's matched
maximum axis error was 42.68 degrees. Later datasets were skipped.

**The full research goal remains open.** All R grid results and S failures
are preserved. The next screen must aggregate every original recorded
trial, using exact streaming derivative covariance statistics rather than
choosing another favorable small context subset.
