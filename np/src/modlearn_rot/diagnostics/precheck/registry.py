from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrecheckFamily:
    module: str
    response_family: str
    implemented: bool
    adapter_module: str


_FAMILIES={
    ("continuous_point",True):PrecheckFamily(
        "continuous","continuous_point",True,
        "modlearn_rot.diagnostics.precheck.adapters.continuous",
    ),
    ("discrete_choice",True):PrecheckFamily(
        "choice","discrete_choice",True,
        "modlearn_rot.diagnostics.precheck.adapters.choice",
    ),
    ("choice_rt_joint",True):PrecheckFamily(
        "choice_rt","choice_rt_joint",False,
        "modlearn_rot.diagnostics.precheck.adapters.choice_rt",
    ),
    ("choice_rt_joint",False):PrecheckFamily(
        "choice_rt","choice_rt_joint",False,
        "modlearn_rot.diagnostics.precheck.adapters.choice_rt",
    ),
    ("questionnaire",False):PrecheckFamily(
        "questionnaire","questionnaire",False,
        "modlearn_rot.diagnostics.precheck.adapters.questionnaire",
    ),
}


def get_precheck_family(response_family:str,has_x:bool)->PrecheckFamily:
    key=(response_family,bool(has_x))
    try:
        return _FAMILIES[key]
    except KeyError as exc:
        raise KeyError(f"No precheck family registered for {key!r}.") from exc
