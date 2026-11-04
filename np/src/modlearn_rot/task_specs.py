"""Central task specifications for the four simulation modules."""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    module: str
    response_family: str
    network_family: str
    encoder_family: str
    decoder_family: str
    has_x: bool
    observation_structure: str
    reconstruction_terms: tuple[str,...]


TASK_SPECS={
    ("continuous_point",True):TaskSpec(
        module="continuous",
        response_family="continuous_point",
        network_family="np_point",
        encoder_family="np_shared_set_gaussian",
        decoder_family="continuous_point",
        has_x=True,
        observation_structure="one continuous response per trial",
        reconstruction_terms=("mse",),
    ),
    ("discrete_choice",True):TaskSpec(
        module="choice",
        response_family="discrete_choice",
        network_family="np_point",
        encoder_family="np_shared_set_gaussian",
        decoder_family="categorical_logits",
        has_x=True,
        observation_structure="one categorical response per trial",
        reconstruction_terms=("cross_entropy",),
    ),
    ("choice_rt_joint",True):TaskSpec(
        module="choice_rt",
        response_family="choice_rt_joint",
        network_family="np_choice_rt",
        encoder_family="np_shared_set_gaussian",
        decoder_family="choice_logits_plus_conditional_rt_flow",
        has_x=True,
        observation_structure="joint choice/RT with X",
        reconstruction_terms=("choice_cross_entropy","rt_negative_log_likelihood"),
    ),
    ("choice_rt_joint",False):TaskSpec(
        module="choice_rt",
        response_family="choice_rt_joint",
        network_family="np_choice_rt",
        encoder_family="np_shared_set_gaussian",
        decoder_family="choice_logits_plus_conditional_rt_flow",
        has_x=False,
        observation_structure="joint choice/RT with x_dim=0",
        reconstruction_terms=("choice_cross_entropy","rt_negative_log_likelihood"),
    ),
    ("questionnaire",False):TaskSpec(
        module="questionnaire",
        response_family="questionnaire",
        network_family="vae_questionnaire",
        encoder_family="questionnaire_vae",
        decoder_family="questionnaire_response_distribution",
        has_x=False,
        observation_structure="subject by item/response vector",
        reconstruction_terms=("family_specific_reconstruction_likelihood",),
    ),
}


def get_task_spec(response_family:str,has_x:bool|None=None)->TaskSpec:
    if has_x is not None:
        key=(response_family,bool(has_x))
        try:
            return TASK_SPECS[key]
        except KeyError as exc:
            raise KeyError(f"Unknown task structure {key!r}.") from exc

    matches=[spec for (family,_),spec in TASK_SPECS.items() if family==response_family]
    if len(matches)==1:
        return matches[0]
    if len(matches)>1:
        raise ValueError(
            f"response_family={response_family!r} has multiple X configurations; "
            "pass has_x explicitly."
        )
    raise KeyError(f"Unknown response family {response_family!r}.")


def get_task_spec_for_model(model)->TaskSpec:
    return get_task_spec(model.response_family,bool(getattr(model,"has_x",True)))
