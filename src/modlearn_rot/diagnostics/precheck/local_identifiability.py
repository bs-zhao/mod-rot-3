import numpy as np


def _information(model,x,b,*,average):
    if hasattr(model,"fisher_information"):
        return np.asarray(
            model.fisher_information(x,b,average=average),dtype=float
        )
    j=model.parameter_jacobian(x,b)
    info=j.T@j
    nv=getattr(model,"noise_std",None)
    if nv is not None:
        info/=float(nv)**2
    return info/max(j.shape[0],1) if average else info


def run_local_identifiability(model,rng,n_parameter_draws=64,n_design=4096):
    x=model.sample_inputs(n_design,rng)
    bs=model.sample_parameters(n_parameter_draws,rng)
    ranks=[]; mins=[]; conds=[]
    mean=np.zeros((model.n_factors,model.n_factors))
    for b in bs:
        info=_information(model,x,b,average=True)
        e=np.linalg.eigvalsh(info)
        ranks.append(np.linalg.matrix_rank(info,tol=1e-8))
        mins.append(e[0])
        conds.append(e[-1]/max(e[0],1e-15))
        mean+=info
    mean/=n_parameter_draws
    ranks=np.asarray(ranks)
    return {
        "mean_fisher":mean,
        "mean_fisher_eigenvalues":np.linalg.eigvalsh(mean),
        "ranks":ranks,
        "min_eigenvalues":np.asarray(mins),
        "condition_numbers":np.asarray(conds),
        "full_rank_fraction":float(np.mean(ranks==model.n_factors)),
        "passed":bool(np.all(ranks==model.n_factors)),
    }
