import numpy as np


def run_validity(model,dataset):
    checks={
        "parameters_valid":bool(model.validate_parameters(dataset.parameters)),
        "inputs_valid":bool(model.validate_inputs(dataset.x)),
        "parameters_finite":bool(np.all(np.isfinite(dataset.parameters))),
        "inputs_finite":bool(np.all(np.isfinite(dataset.x))),
        "observations_finite":bool(np.all(np.isfinite(dataset.y))),
        "predictive_object_finite":bool(
            np.all(np.isfinite(dataset.predictive_true))
        ),
        "subject_count_matches":dataset.parameters.shape[0]==model.n_subjects,
        "factor_count_matches":dataset.parameters.shape[1]==model.n_factors,
        "trial_count_sum_matches":int(dataset.trial_counts.sum())==len(dataset.y),
    }
    if model.response_family=="discrete_choice":
        y=np.asarray(dataset.y)
        checks["choice_labels_valid"]=bool(
            np.all(np.equal(y,np.round(y)))
            and np.all((y>=0)&(y<int(getattr(model,"num_choices",0))))
        )
        p=np.asarray(dataset.predictive_true,dtype=float)
        checks["predictive_probability_valid"]=bool(
            np.all((p>=0.0)&(p<=1.0))
        )
    checks["passed"]=bool(all(checks.values()))
    return checks
