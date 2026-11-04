import numpy as np
from ...generators.dataset import iter_subject_slices


def _r2(y,p):
    ssr=float(np.sum((y-p)**2))
    sst=float(np.sum((y-y.mean())**2))
    return 1.0-ssr/sst if sst>0 else float("nan")


def _binary_choice_scores(model,dataset,oracle_estimates=None):
    y=np.asarray(dataset.y,dtype=float).reshape(-1)
    p=np.clip(
        np.asarray(dataset.predictive_true,dtype=float).reshape(-1),
        1e-12,1.0-1e-12,
    )
    prevalence=float(np.mean(y))
    null_p=np.full_like(p,prevalence)

    def brier(prob):
        return float(np.mean((y-prob)**2))

    def log_loss(prob):
        prob=np.clip(prob,1e-12,1.0-1e-12)
        return float(-np.mean(y*np.log(prob)+(1.0-y)*np.log1p(-prob)))

    brier_true=brier(p)
    brier_null=brier(null_p)
    log_true=log_loss(p)
    log_null=log_loss(null_p)
    classification_accuracy=float(np.mean((p>=0.5)==(y>=0.5)))
    bayes_accuracy_expected=float(np.mean(np.maximum(p,1.0-p)))
    null_accuracy=float(max(prevalence,1.0-prevalence))

    # Oracle-fitted probability surface: this assesses finite-trial recoverability
    # in the metric the decoder ultimately needs to reproduce.
    oracle=np.empty_like(p)
    if oracle_estimates is None:
        oracle_estimates=np.empty_like(dataset.parameters,dtype=float)
        for sid,left,right in iter_subject_slices(dataset):
            oracle_estimates[sid]=model.oracle_fit(
                dataset.x[left:right],dataset.y[left:right]
            )
    for sid,left,right in iter_subject_slices(dataset):
        oracle[left:right]=np.asarray(
            model.predictive_object(
                dataset.x[left:right],oracle_estimates[sid]
            )
        ).reshape(-1)
    oracle=np.clip(oracle,1e-12,1.0-1e-12)
    brier_oracle=brier(oracle)
    log_oracle=log_loss(oracle)

    brier_skill=1.0-brier_true/max(brier_null,1e-15)
    log_skill=1.0-log_true/max(log_null,1e-15)
    passed=(
        np.isfinite(brier_true)
        and np.isfinite(log_true)
        and brier_true<brier_null
        and log_true<log_null
        and bayes_accuracy_expected>null_accuracy+0.05
        and brier_oracle<brier_null
        and log_oracle<log_null
    )
    return {
        "metric_family":"binary_choice",
        "choice_rate":prevalence,
        "bayes_accuracy_expected":bayes_accuracy_expected,
        "realized_true_probability_accuracy":classification_accuracy,
        "null_accuracy":null_accuracy,
        "brier_true_probability":brier_true,
        "brier_oracle_estimated_probability":brier_oracle,
        "brier_null":brier_null,
        "brier_skill_over_null":brier_skill,
        "log_loss_true_probability":log_true,
        "log_loss_oracle_estimated_probability":log_oracle,
        "log_loss_null":log_null,
        "log_loss_skill_over_null":log_skill,
        "passed":bool(passed),
    }


def run_predictive_ceiling(model,dataset,oracle_estimates=None):
    if model.response_family=="discrete_choice":
        if int(getattr(model,"num_choices",0))==2:
            return _binary_choice_scores(model,dataset,oracle_estimates=oracle_estimates)
        raise NotImplementedError(
            "Predictive-ceiling precheck is not yet implemented for "
            f"num_choices={getattr(model,'num_choices',None)}."
        )

    empirical=_r2(dataset.y,dataset.predictive_true)
    theoretical=model.theoretical_prediction_ceiling()
    oracle=np.empty_like(dataset.y,dtype=float)
    if oracle_estimates is None:
        oracle_estimates=np.empty_like(dataset.parameters,dtype=float)
        for sid,left,right in iter_subject_slices(dataset):
            oracle_estimates[sid]=model.oracle_fit(
                dataset.x[left:right],dataset.y[left:right]
            )
    for sid,left,right in iter_subject_slices(dataset):
        oracle[left:right]=model.predictive_object(
            dataset.x[left:right],oracle_estimates[sid]
        )
    oracle_r2=_r2(dataset.y,oracle)
    passed=np.isfinite(empirical)
    if theoretical is not None:
        passed=passed and abs(empirical-theoretical)<0.03
    return {
        "metric_family":"continuous_r2",
        "theoretical_r2":theoretical,
        "empirical_true_mean_r2":empirical,
        "oracle_estimated_parameter_r2":oracle_r2,
        "passed":bool(passed),
    }
