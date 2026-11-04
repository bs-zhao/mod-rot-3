"""Single-response Neural Process models."""

from .continuous import ContinuousNPConfig,ContinuousNeuralProcess
from .discrete import DiscreteNPConfig,DiscreteNeuralProcess

__all__=[
    "ContinuousNPConfig","ContinuousNeuralProcess",
    "DiscreteNPConfig","DiscreteNeuralProcess",
]
