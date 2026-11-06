# Complete scalar metric catalogue

All quantities are descriptive. A higher or lower value is not automatically a better rotation. The experiment computes 25 per-axis families and 70 scalar summaries; aggregates/proxies are not independent pieces of evidence.

For axis statistics s_i: RMS = sqrt(mean_i s_i^2). Axis entropy = -sum p_i log p_i / log K, p_i=s_i^2/sum s_j^2. NA is retained if all axes are undefined.

| ID | Scalar output | Definition / parent estimator |
|---|---|---|
| M001 | `pearson__axis_rms` | Centered linear correlation between one U axis and Y. Aggregate: axis_rms. |
| M002 | `pearson__axis_entropy` | Centered linear correlation between one U axis and Y. Aggregate: axis_entropy. |
| M003 | `spearman__axis_rms` | Pearson correlation of marginal average ranks. Aggregate: axis_rms. |
| M004 | `spearman__axis_entropy` | Pearson correlation of marginal average ranks. Aggregate: axis_entropy. |
| M005 | `kendall__axis_rms` | Kendall tau-b with response ties retained. Aggregate: axis_rms. |
| M006 | `kendall__axis_entropy` | Kendall tau-b with response ties retained. Aggregate: axis_entropy. |
| M007 | `chatterjee_xi__axis_rms` | Directional Chatterjee xi with max/min response ranks and deterministic randomized X tie order. Aggregate: axis_rms. |
| M008 | `chatterjee_xi__axis_entropy` | Directional Chatterjee xi with max/min response ranks and deterministic randomized X tie order. Aggregate: axis_entropy. |
| M009 | `rdc__axis_rms` | Largest regularized canonical correlation of rank-copula random sine features; 10 features, fixed seeds, ridge 1e-4. Aggregate: axis_rms. |
| M010 | `rdc__axis_entropy` | Largest regularized canonical correlation of rank-copula random sine features; 10 features, fixed seeds, ridge 1e-4. Aggregate: axis_entropy. |
| M011 | `distance_correlation_squared__axis_rms` | Biased normalized inner product of doubly centered pairwise distance matrices (squared dCor). Aggregate: axis_rms. |
| M012 | `distance_correlation_squared__axis_entropy` | Biased normalized inner product of doubly centered pairwise distance matrices (squared dCor). Aggregate: axis_entropy. |
| M013 | `normalized_hsic__axis_rms` | Centered RBF kernel alignment, i.e. HSIC divided by kernel self-norms; median squared-distance bandwidth. Aggregate: axis_rms. |
| M014 | `normalized_hsic__axis_entropy` | Centered RBF kernel alignment, i.e. HSIC divided by kernel self-norms; median squared-distance bandwidth. Aggregate: axis_entropy. |
| M015 | `binned_eta_squared__axis_rms` | Within-quantile-bin conditional-mean variance divided by Y variance; in-sample descriptive estimator. Aggregate: axis_rms. |
| M016 | `binned_eta_squared__axis_entropy` | Within-quantile-bin conditional-mean variance divided by Y variance; in-sample descriptive estimator. Aggregate: axis_entropy. |
| M017 | `binned_mi_nats__axis_rms` | Plug-in mutual information of quantile bins, in nats; binary Y retains two categories. Aggregate: axis_rms. |
| M018 | `binned_mi_nats__axis_entropy` | Plug-in mutual information of quantile bins, in nats; binary Y retains two categories. Aggregate: axis_entropy. |
| M019 | `adaptive_grid_nmi_proxy__axis_rms` | Maximum normalized discrete MI over 2/3/4/6 matched marginal bin counts; not MIC/MICe. Aggregate: axis_rms. |
| M020 | `adaptive_grid_nmi_proxy__axis_entropy` | Maximum normalized discrete MI over 2/3/4/6 matched marginal bin counts; not MIC/MICe. Aggregate: axis_entropy. |
| M021 | `partial_pearson_given_X__axis_rms` | Correlation of OLS residuals of U_i and Y after conditioning linearly on X. Aggregate: axis_rms. |
| M022 | `partial_pearson_given_X__axis_entropy` | Correlation of OLS residuals of U_i and Y after conditioning linearly on X. Aggregate: axis_entropy. |
| M023 | `partial_pearson_given_otherU_X__axis_rms` | Correlation of OLS residuals after conditioning on [X,U_-i]. Aggregate: axis_rms. |
| M024 | `partial_pearson_given_otherU_X__axis_entropy` | Correlation of OLS residuals after conditioning on [X,U_-i]. Aggregate: axis_entropy. |
| M025 | `partial_spearman_given_X__axis_rms` | Partial linear correlation in marginal rank coordinates; not general conditional dependence. Aggregate: axis_rms. |
| M026 | `partial_spearman_given_X__axis_entropy` | Partial linear correlation in marginal rank coordinates; not general conditional dependence. Aggregate: axis_entropy. |
| M027 | `pdcor_given_X__axis_rms` | Cosine of U-centered distance matrices after projection off the U-centered X distance matrix. Aggregate: axis_rms. |
| M028 | `pdcor_given_X__axis_entropy` | Cosine of U-centered distance matrices after projection off the U-centered X distance matrix. Aggregate: axis_entropy. |
| M029 | `pdcor_given_otherU_X__axis_rms` | Same residual distance correlation, conditioning on joint Euclidean [X,U_-i]. Aggregate: axis_rms. |
| M030 | `pdcor_given_otherU_X__axis_entropy` | Same residual distance correlation, conditioning on joint Euclidean [X,U_-i]. Aggregate: axis_entropy. |
| M031 | `codec_given_X__axis_rms` | Azadkia-Chatterjee rank/min nearest-neighbor conditional dependence of Y on U_i given X. Aggregate: axis_rms. |
| M032 | `codec_given_X__axis_entropy` | Azadkia-Chatterjee rank/min nearest-neighbor conditional dependence of Y on U_i given X. Aggregate: axis_entropy. |
| M033 | `codec_given_otherU_X__axis_rms` | CODEC of Y on U_i given [X,U_-i]; NA when empirical denominator is zero. Aggregate: axis_rms. |
| M034 | `codec_given_otherU_X__axis_entropy` | CODEC of Y on U_i given [X,U_-i]; NA when empirical denominator is zero. Aggregate: axis_entropy. |
| M035 | `cmi_knn_given_X__axis_rms` | Continuous Frenzel-Pompe kNN CMI with Chebyshev product neighborhoods and k=5; binary Y is NA. Aggregate: axis_rms. |
| M036 | `cmi_knn_given_X__axis_entropy` | Continuous Frenzel-Pompe kNN CMI with Chebyshev product neighborhoods and k=5; binary Y is NA. Aggregate: axis_entropy. |
| M037 | `cmi_knn_given_otherU_X__axis_rms` | Same continuous estimator conditioning on [X,U_-i]; binary Y is NA. Aggregate: axis_rms. |
| M038 | `cmi_knn_given_otherU_X__axis_entropy` | Same continuous estimator conditioning on [X,U_-i]; binary Y is NA. Aggregate: axis_entropy. |
| M039 | `coarsened_X_cmi_nats__axis_rms` | Plug-in CMI within 3x3 quantile strata of first two X PCs; not full-X CMI. Aggregate: axis_rms. |
| M040 | `coarsened_X_cmi_nats__axis_entropy` | Plug-in CMI within 3x3 quantile strata of first two X PCs; not full-X CMI. Aggregate: axis_entropy. |
| M041 | `UX_interaction_information_proxy__axis_rms` | MI((binned U_i,coarsened X);Y)-MI(U_i;Y)-MI(coarsened X;Y), signed; not PID synergy. Aggregate: axis_rms. |
| M042 | `UX_interaction_information_proxy__axis_entropy` | MI((binned U_i,coarsened X);Y)-MI(U_i;Y)-MI(coarsened X;Y), signed; not PID synergy. Aggregate: axis_entropy. |
| M043 | `full_U_dcor_squared_control` | Squared distance correlation between the complete U vector and Y; exact orthogonal negative control. |
| M044 | `functional_linear_cv_r2_control` | Grouped-CV ridge R2 for complete [1,U] x degree-2 X features; same complete model under U rotation. |
| M045 | `functional_quadratic_cv_r2_control` | Grouped-CV ridge R2 for complete degree-2 U x degree-2 X features, with symmetric orthonormal packing. |
| M046 | `full_rbf_cv_r2_control` | Grouped-CV isotropic Euclidean RBF kernel ridge over [U,X]; exact orthogonal negative control. |
| M047 | `proxy_module_code` | Sum log(1+g_m/tau), g_m=RMS_X of each nonconstant-U coefficient function / common L2 energy; tau=.02/sqrt(number of U modules). |
| M048 | `proxy_module_group_l1` | Sum of normalized coefficient-function RMS across all linear and pure/mixed quadratic U modules. |
| M049 | `proxy_linear_U_module_code` | Same module code restricted to degree-1 U modules, using the same common denominator/floor. |
| M050 | `proxy_quadratic_U_module_code` | Same module code restricted to all degree-2 U modules; pure and mixed terms are included. |
| M051 | `proxy_hessian_offdiag_fraction` | Mean squared off-diagonal U-Hessian / mean full squared Frobenius norm of the quadratic proxy. |
| M052 | `proxy_hessian_diagonal_fraction` | Complementary pure-diagonal squared Hessian energy fraction. |
| M053 | `proxy_hessian_group_l1` | Sum RMS of all full Hessian entries divided by total Hessian Frobenius RMS; symmetric mixed entries counted twice as explicitly implemented. |
| M054 | `proxy_hessian_energy_control` | Full U-Hessian mean squared Frobenius norm; exact rotation invariant. |
| M055 | `proxy_gradient_energy_control` | Mean squared full U-gradient norm; exact rotation invariant. |
| M056 | `proxy_axis_sensitivity_entropy` | Entropy of per-axis mean squared proxy gradient energies. |
| M057 | `proxy_XU_energy_control` | Full mixed X/U Jacobian mean squared Frobenius norm; exact rotation invariant. |
| M058 | `proxy_XU_group_l1` | Sum RMS across all X/U mixed entries divided by full mixed Frobenius RMS. |
| M059 | `proxy_XU_channel_entropy` | Average across X of entropy of axis shares of mixed-derivative second moments. |
| M060 | `proxy_XU_channel_geomin` | Average across X of geometric mean of axis energy shares + 1e-6. |
| M061 | `proxy_effect_profile_effective_rank_mean` | Mean per-axis entropy effective rank of proxy gradient profiles over common observed X, across distinct subject states. |
| M062 | `gam_cv_r2` | Subject-grouped CV R2 for additive cubic B-spline ridge of all U and X. |
| M063 | `gam_increment_over_X_cv_r2` | CV SSE reduction of additive U+X model versus X-only additive model, divided by SST. |
| M064 | `gam_partial_cv_r2__axis_sum` | Sum over U_i of (SSE_without_i-SSE_full)/SSE_without_i, held-out subject predictions. |
| M065 | `gam_partial_cv_r2__axis_min` | Minimum per-axis CV partial R2. |
| M066 | `gam_semipartial_cv_r2__axis_sum` | Sum over U_i of (SSE_without_i-SSE_full)/SST. |
| M067 | `gam_semipartial_cv_r2__axis_min` | Minimum per-axis CV semipartial R2. |
| M068 | `axis_excess_kurtosis_squared_sum` | Sum squared empirical excess kurtosis over U axes, using one coordinate per subject; ignores Y. |
| M069 | `axis_fourth_moment_sum` | Sum empirical fourth centered moments of U axes; ignores Y. |
| M070 | `U_total_variance_control` | Mean squared norm of centered full U over subjects; exact orthogonal negative control, ignores Y. |

Proxy derivatives are analytic derivatives of a low-order UXY ridge regression, not of the trained decoder. Their predictive approximation scores must be read alongside them. Nearest-neighbor estimates use one row per subject. No p-values or universal axis-recovery claims are made.
