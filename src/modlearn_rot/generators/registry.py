"""Only the original C1/C2/C3 binary-choice generators are registered."""
from .choice.c1_linear_logit import ChoiceC1LinearLogit
from .choice.c2_shared_x_quadratic import ChoiceC2SharedXQuadratic
from .choice.c3_multi_nonlinear_logit import ChoiceC3MultiNonlinearLogit
_MODEL_REGISTRY={"C1":ChoiceC1LinearLogit,"C2":ChoiceC2SharedXQuadratic,"C3":ChoiceC3MultiNonlinearLogit}
def get_model(model_id: str):
    key=model_id.upper()
    if key not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model {model_id!r}; options: {tuple(sorted(_MODEL_REGISTRY))}")
    return _MODEL_REGISTRY[key]()
def list_models() -> tuple[str,...]:
    return tuple(sorted(_MODEL_REGISTRY))
