import numpy as np
from ...generators.dataset import iter_subject_slices


def run_design_information(model,dataset):
    ranks=np.empty(model.n_subjects,dtype=int)
    mins=np.empty(model.n_subjects)
    conds=np.empty(model.n_subjects)
    for sid,left,right in iter_subject_slices(dataset):
        if hasattr(model,"fisher_information"):
            info=np.asarray(
                model.fisher_information(
                    dataset.x[left:right],dataset.parameters[sid],average=False
                ),
                dtype=float,
            )
        else:
            j=model.parameter_jacobian(
                dataset.x[left:right],dataset.parameters[sid]
            )
            info=j.T@j
            nv=getattr(model,"noise_std",None)
            if nv is not None:
                info/=float(nv)**2
        e=np.linalg.eigvalsh(info)
        ranks[sid]=np.linalg.matrix_rank(info,tol=1e-8)
        mins[sid]=e[0]
        conds[sid]=e[-1]/max(e[0],1e-15)
    return {
        "information_definition":(
            "Fisher information" if hasattr(model,"fisher_information")
            else "Gaussian/derivative information"
        ),
        "ranks":ranks,
        "min_eigenvalues":mins,
        "condition_numbers":conds,
        "full_rank_fraction":float(np.mean(ranks==model.n_factors)),
        "median_min_eigenvalue":float(np.median(mins)),
        "p05_min_eigenvalue":float(np.quantile(mins,0.05)),
        "median_condition_number":float(np.median(conds)),
        "passed":bool(np.all(ranks==model.n_factors)),
    }
