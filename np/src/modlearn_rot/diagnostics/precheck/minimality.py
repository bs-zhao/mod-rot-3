import numpy as np


def run_minimality(model, rng, n_parameter_draws=512, n_design=2048):
    x = model.sample_inputs(n_design, rng)
    bs = model.sample_parameters(n_parameter_draws, rng)
    local = np.asarray([np.linalg.matrix_rank(model.parameter_jacobian(x,b), tol=1e-8) for b in bs[:64]])
    f = np.vstack([np.asarray(model.predictive_object(x,b)).reshape(1,-1) for b in bs])
    fc = f - f.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(fc, compute_uv=False)
    tol = max(fc.shape) * np.finfo(float).eps * sv[0]
    return {
        "local_jacobian_ranks": local,
        "minimum_local_rank": int(local.min()),
        "linear_function_rank": int(np.sum(sv > tol)),
        "function_singular_values": sv[:min(32, sv.size)],
        "note": "Function PCA rank is descriptive for nonlinear-in-parameter models; local Jacobian rank is primary.",
        "passed": bool(np.all(local == model.n_factors)),
    }
