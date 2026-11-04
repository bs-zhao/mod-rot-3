"""Response-family balance checks used before representation learning."""

from __future__ import annotations

import numpy as np


def _binary_choice_balance(model,dataset):
    y=np.asarray(dataset.y).reshape(-1)
    p=np.asarray(dataset.predictive_true,dtype=float).reshape(-1)
    valid_y=bool(np.all((y==0)|(y==1)))
    valid_p=bool(np.all(np.isfinite(p)) and np.all((p>=0)&(p<=1)))

    choice_rate=float(np.mean(y))
    expected_choice_rate=float(np.mean(p))
    probability_sd=float(np.std(p))
    bayes_accuracy=float(np.mean(np.maximum(p,1.0-p)))
    saturation_fraction=float(np.mean((p<0.02)|(p>0.98)))
    q=np.quantile(p,[0.01,0.05,0.25,0.5,0.75,0.95,0.99])

    passed=(
        valid_y
        and valid_p
        and abs(expected_choice_rate-0.5)<0.02
        and abs(choice_rate-0.5)<0.03
        and probability_sd>0.10
        and 0.60<bayes_accuracy<0.85
        and saturation_fraction<0.05
    )
    return {
        "num_choices":2,
        "choice_rate":choice_rate,
        "expected_choice_rate":expected_choice_rate,
        "probability_sd":probability_sd,
        "bayes_accuracy_expected":bayes_accuracy,
        "saturation_fraction_p_lt_002_or_gt_098":saturation_fraction,
        "probability_quantiles_01_05_25_50_75_95_99":q,
        "valid_binary_observations":valid_y,
        "valid_probabilities":valid_p,
        "passed":bool(passed),
    }


def run_response_balance(model,dataset):
    if model.response_family=="discrete_choice":
        if int(getattr(model,"num_choices",0))==2:
            return _binary_choice_balance(model,dataset)
        raise NotImplementedError(
            "Response-balance precheck is not yet implemented for "
            f"num_choices={getattr(model,'num_choices',None)}."
        )
    return {
        "not_applicable":True,
        "passed":True,
    }
