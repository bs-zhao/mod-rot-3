import numpy as np


def run_information_balance(model,rng,n_parameter_draws=128,n_design=4096):
    x=model.sample_inputs(n_design,rng)
    bs=model.sample_parameters(n_parameter_draws,rng)
    info=np.zeros(model.n_factors)
    for b in bs:
        if hasattr(model,"fisher_information"):
            matrix=np.asarray(
                model.fisher_information(x,b,average=True),dtype=float
            )
            info+=np.diag(matrix)
        else:
            j=model.parameter_jacobian(x,b)
            info+=np.mean(j**2,axis=0)
    info/=n_parameter_draws
    prior_sd=np.std(
        model.sample_parameters(20000,rng),axis=0,ddof=1
    )
    scaled=info*prior_sd**2
    normalized=scaled/scaled.sum()
    positive=scaled[scaled>0]
    ratio=float(positive.max()/positive.min())
    return {
        "information_definition":(
            "Fisher diagonal" if hasattr(model,"fisher_information")
            else "mean squared predictive derivative"
        ),
        "raw_information":info,
        "prior_sd":prior_sd,
        "prior_scaled_information":scaled,
        "normalized_contribution":normalized,
        "max_min_ratio":ratio,
        "passed":bool(ratio<4.0),
    }
