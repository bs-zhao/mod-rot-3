"""Machine-readable registry of implemented local Hessian indicators.

All metrics are generated from explicit local derivative tensors and/or
controlled one-axis sweeps. This is NOT just a wish list.
"""
from __future__ import annotations


def classify(name):
    if name.endswith("_loss"):
        direction="minimize"
    elif name.endswith("_diagnostic"):
        direction="diagnostic_only"
    else:
        direction="descriptive"
    if name.startswith("sweep_"):
        family="controlled_single_Z_sweep"
    elif name.startswith("raw_"):
        family="unstandardized_physical_unit"
    elif name.startswith("zz_self"):
        family="ZZ_self_higher_order"
    elif name.startswith("zz_other") or name.startswith("zz_mixed") or name.startswith("zz_Z"):
        family="ZZ_cross_Z_nonlinearity"
    elif name.startswith(("zz_X_","zz_diagonal_X","zz_offdiag_X")):
        family="ZZ_X_modulation"
    elif name.startswith("zz_"):
        family="ZZ_local_structure_and_sparsity"
    elif name.startswith("zx_local_row"):
        family="ZX_pointwise_row_support"
    elif name.startswith("zx_own"):
        family="ZX_self_Z_X_modulation"
    elif name.startswith("zx_other"):
        family="ZX_other_Z_modulation"
    elif name.startswith("zx_"):
        family="ZX_local_structure_and_sparsity"
    else:family="additional"
    if "fourth" in name or "zzxx" in name or "zxxx" in name:
        order="fourth"
    elif "third" in name or "higher_order" in name or "modulation" in name:
        order="third_or_higher"
    elif "sweep" in name and "quadratic" in name:
        order="function_value_quadratic_residual"
    elif "sweep" in name:
        order="function_value_or_curvature_scan"
    else:order="second_local_hessian"
    if "tail" in name:
        description="Sorted absolute effects beyond top-k, with configurable k"
    elif "capped" in name or "softl0" in name or "logsum" in name:
        description="Saturated/soft-L0 surrogate preserving tiny residual effects"
    elif "support" in name:
        description="Adaptive number of active effects plus residual energy"
    elif "curvature_variance" in name:
        description="Variation of H_ii as only U_i is varied, other U and X fixed"
    elif "approx_relative_mse" in name:
        description="Polynomial degree d fit of a controlled one-axis response curve"
    elif "X_modulation" in name:
        description="Sensitivity of a ZZ curvature component to X"
    elif "other_Z" in name:
        description="Higher-order dependence involving a different Z coordinate"
    elif "self_z" in name or "self_Z" in name:
        description="Higher-order dependence along its own Z coordinate"
    elif "OOM" in name or "OOD" in name or "Mahalanobis" in name:
        description="Extrapolation risk diagnostic, NOT a simplicity objective"
    elif "rms" in name:
        description="Local derivative root-mean-square energy"
    elif "cancellation" in name:
        description="Cancellation when averaging local Hessians"
    else:
        description="Derivative, smoothness, effective-support or distribution statistic"
    return dict(metric=name,optimal_direction=direction,
                indicator_family=family,derivative_order=order,
                interpretation=description)


def describe_all(metric_names):
    return [classify(name) for name in sorted(metric_names)]
