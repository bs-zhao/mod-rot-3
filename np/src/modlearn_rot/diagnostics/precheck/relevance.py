import numpy as np


def run_relevance(model, rng, n_parameter_draws=64, n_design=4096):
    x = model.sample_inputs(n_design, rng)
    bs = model.sample_parameters(n_parameter_draws, rng)
    s = np.zeros(model.n_factors)
    for b in bs:
        j = model.parameter_jacobian(x, b)
        s += np.mean(j**2, axis=0)
    s /= n_parameter_draws
    return {"sensitivity": s, "min_sensitivity": float(s.min()), "passed": bool(np.all(s > 1e-8))}
