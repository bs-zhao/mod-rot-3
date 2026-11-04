"""Generated-dataset containers, generation, and serialization.

These utilities live with the generators because they only describe simulation
artifacts.  Runtime data themselves remain under outputs/ and are never synced
to Git.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..utils.seeds import make_rng_streams


@dataclass
class GeneratedDataset:
    """Trialwise simulation data. X may have zero columns."""

    parameters: np.ndarray
    trial_counts: np.ndarray
    subject_index: np.ndarray
    trial_index: np.ndarray
    x: np.ndarray
    y: np.ndarray
    predictive_true: np.ndarray


@dataclass
class QuestionnaireGeneratedDataset:
    """No-X subject-by-item questionnaire simulation data."""

    parameters: np.ndarray
    responses: np.ndarray
    mask: np.ndarray
    predictive_true: np.ndarray


@dataclass
class ObservedDataset:
    """Truth-free trialwise view used by training/rotation."""

    trial_counts: np.ndarray
    subject_index: np.ndarray
    trial_index: np.ndarray
    x: np.ndarray
    y: np.ndarray


@dataclass
class ObservedQuestionnaireDataset:
    responses: np.ndarray
    mask: np.ndarray


def _trial_array(value, n_trials: int, name: str) -> np.ndarray:
    arr=np.asarray(value)
    if arr.ndim==0 or arr.shape[0]!=int(n_trials):
        raise ValueError(
            f"{name} must have first dimension n_trials={n_trials}; got {arr.shape}."
        )
    return arr



def iter_subject_slices(dataset):
    """Yield (subject_id, left, right) for contiguous per-person trial blocks.

    Dataset generation concatenates trials in subject order. This enables
    O(total_trials) precheck access instead of building one O(total_trials)
    boolean mask per person (which was O(n_subjects * total_trials)).
    Validate the layout once before using the zero-copy contiguous slices.
    """
    counts=np.asarray(dataset.trial_counts,dtype=np.int64)
    ids=np.asarray(dataset.subject_index,dtype=np.int64)
    if counts.ndim!=1 or len(ids)!=int(counts.sum()):
        raise ValueError("Subject trial count metadata is inconsistent")
    if np.any(counts<1):
        raise ValueError("Every subject must have positive trial count")
    boundaries=np.concatenate(([0],np.cumsum(counts)))
    # Checking grouped order only at boundary indices would not catch an
    # accidentally interleaved subject, so verify each block in a single
    # vectorized pass using np.repeat.
    if not np.array_equal(ids,np.repeat(np.arange(len(counts)),counts)):
        raise ValueError("Subject IDs are not concatenated in trial order")
    for sid in range(len(counts)):
        yield sid,int(boundaries[sid]),int(boundaries[sid+1])


def generate_trialwise_dataset(model, seed: int) -> GeneratedDataset:
    rngs=make_rng_streams(seed,("parameters","trial_counts","inputs","observations"))
    parameters=model.sample_parameters(model.n_subjects,rngs["parameters"])
    counts=rngs["trial_counts"].integers(
        model.trial_count_min,model.trial_count_max+1,size=model.n_subjects
    )
    # Models may declare a strict hard cap for real simulated observations.
    # Large sample_inputs(n_design) calls in information diagnostics do NOT
    # represent participant data and remain independent of this check.
    hard_cap=getattr(model,"max_trials_per_subject",None)
    if hard_cap is not None and (
        int(np.max(counts))>int(hard_cap)
        or int(model.trial_count_max)>int(hard_cap)
    ):
        raise ValueError(
            f"{model.model_id} trial count exceeds hard per-person cap "
            f"{int(hard_cap)}"
        )
    subjects=[]; trials=[]; xs=[]; ys=[]; fs=[]
    for sid,(b,n) in enumerate(zip(parameters,counts,strict=True)):
        n=int(n)
        x=_trial_array(model.sample_inputs(n,rngs["inputs"]),n,"X")
        if x.ndim==1:
            x=x[:,None]
        f=_trial_array(model.predictive_object(x,b),n,"predictive_object")
        y=_trial_array(model.sample_observations(x,b,rngs["observations"]),n,"Y")
        subjects.append(np.full(n,sid,dtype=np.int64))
        trials.append(np.arange(n,dtype=np.int64))
        xs.append(x); ys.append(y); fs.append(f)
    return GeneratedDataset(
        parameters=np.asarray(parameters),
        trial_counts=np.asarray(counts,dtype=np.int64),
        subject_index=np.concatenate(subjects),
        trial_index=np.concatenate(trials),
        x=np.concatenate(xs,axis=0),
        y=np.concatenate(ys,axis=0),
        predictive_true=np.concatenate(fs,axis=0),
    )


def generate_questionnaire_dataset(model, seed: int) -> QuestionnaireGeneratedDataset:
    rngs=make_rng_streams(seed,("parameters","observations","mask"))
    parameters=np.asarray(model.sample_parameters(model.n_subjects,rngs["parameters"]))
    responses=np.asarray(model.sample_responses(parameters,rngs["observations"]))
    if responses.shape[0]!=model.n_subjects:
        raise ValueError("Questionnaire responses must have first dimension n_subjects.")
    predictive=np.asarray(model.predictive_object(parameters))
    if predictive.shape[0]!=model.n_subjects:
        raise ValueError("Questionnaire predictive object must have first dimension n_subjects.")
    if hasattr(model,"sample_response_mask"):
        mask=np.asarray(model.sample_response_mask(responses,rngs["mask"]),dtype=bool)
    else:
        mask=np.isfinite(responses)
    if mask.shape!=responses.shape:
        raise ValueError("Questionnaire response mask must match responses shape.")
    return QuestionnaireGeneratedDataset(
        parameters=parameters,
        responses=responses,
        mask=mask,
        predictive_true=predictive,
    )


def generate_dataset(model, seed: int):
    if getattr(model,"data_layout","trialwise")=="subject_by_item":
        return generate_questionnaire_dataset(model,seed)
    return generate_trialwise_dataset(model,seed)


def summarize_dataset(dataset) -> dict:
    if isinstance(dataset,GeneratedDataset):
        x=np.asarray(dataset.x); y=np.asarray(dataset.y)
        no_x=(x.ndim==2 and x.shape[1]==0)
        out={
            "dataset_kind":"trialwise_no_x" if no_x else "trialwise",
            "n_subjects":int(dataset.parameters.shape[0]),
            "total_trials":int(len(dataset.subject_index)),
            "trial_count_min_observed":int(dataset.trial_counts.min()),
            "trial_count_max_observed":int(dataset.trial_counts.max()),
            "trial_count_mean":float(dataset.trial_counts.mean()),
            "parameter_mean":np.mean(dataset.parameters,axis=0),
            "parameter_sd":np.std(dataset.parameters,axis=0,ddof=1),
            "x_shape":list(x.shape),
            "y_shape":list(y.shape),
            "y_mean":np.mean(y,axis=0),
            "y_sd":np.std(y,axis=0,ddof=1),
        }
        if not no_x:
            out["x_mean"]=np.mean(x,axis=0)
            out["x_sd"]=np.std(x,axis=0,ddof=1)
        return out
    if isinstance(dataset,QuestionnaireGeneratedDataset):
        r=np.asarray(dataset.responses)
        return {
            "dataset_kind":"questionnaire",
            "n_subjects":int(dataset.parameters.shape[0]),
            "response_shape":list(r.shape),
            "observed_fraction":float(np.mean(dataset.mask)),
            "parameter_mean":np.mean(dataset.parameters,axis=0),
            "parameter_sd":np.std(dataset.parameters,axis=0,ddof=1),
            "response_mean":np.nanmean(np.where(dataset.mask,r,np.nan),axis=0),
            "response_sd":np.nanstd(np.where(dataset.mask,r,np.nan),axis=0,ddof=1),
        }
    raise TypeError(f"Unsupported dataset type {type(dataset).__name__}.")


def save_dataset(dataset, path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(dataset.__dict__)
    payload["__dataset_kind__"]=np.asarray(
        "trialwise" if isinstance(dataset,GeneratedDataset) else "questionnaire"
    )
    np.savez_compressed(path,**payload)


def load_dataset(path):
    with np.load(Path(path),allow_pickle=False) as d:
        keys=set(d.files)
        payload={k:d[k] for k in d.files if not k.startswith("__")}
    if {"trial_counts","subject_index","trial_index","x","y"}.issubset(keys):
        return GeneratedDataset(**payload)
    if {"responses","mask"}.issubset(keys):
        return QuestionnaireGeneratedDataset(**payload)
    raise ValueError(f"Unrecognized generated dataset layout in {path}.")


def load_observed_dataset(path) -> ObservedDataset:
    with np.load(Path(path),allow_pickle=False) as d:
        required=("trial_counts","subject_index","trial_index","x","y")
        missing=[name for name in required if name not in d.files]
        if missing:
            raise TypeError(
                f"Trialwise loader missing {missing}; questionnaire data use "
                "load_observed_questionnaire_dataset."
            )
        return ObservedDataset(
            trial_counts=d["trial_counts"],
            subject_index=d["subject_index"],
            trial_index=d["trial_index"],
            x=d["x"],
            y=d["y"],
        )


def load_observed_questionnaire_dataset(path) -> ObservedQuestionnaireDataset:
    with np.load(Path(path),allow_pickle=False) as d:
        required=("responses","mask")
        missing=[name for name in required if name not in d.files]
        if missing:
            raise TypeError(f"Questionnaire loader missing {missing}.")
        return ObservedQuestionnaireDataset(
            responses=d["responses"],
            mask=d["mask"],
        )
