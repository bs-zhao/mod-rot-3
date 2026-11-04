import numpy as np


def run_functional_distinguishability(model,rng,n_parameter_draws=64,n_design=4096):
    x=model.sample_inputs(n_design,rng)
    bs=model.sample_parameters(n_parameter_draws,rng)
    gram=np.zeros((model.n_factors,model.n_factors))
    for b in bs:
        j=model.parameter_jacobian(x,b)
        scale=np.sqrt(np.mean(j**2,axis=0))
        jn=j/np.maximum(scale,1e-15)
        gram += (jn.T@jn)/j.shape[0]
    gram/=n_parameter_draws
    off=gram-np.diag(np.diag(gram))
    m=float(np.max(np.abs(off)))
    return {"normalized_derivative_gram":gram,"max_abs_offdiag":m,"passed":bool(m<0.98)}
