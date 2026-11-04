from __future__ import annotations

from abc import ABC,abstractmethod
from dataclasses import asdict,dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    model_version: str
    group: str
    response_family: str
    n_factors: int
    parameter_names: tuple[str,...]
    n_subjects: int
    trial_count_min: int
    trial_count_max: int
    equation: str
    description: str
    topology_tags: tuple[str,...]
    functional_form_tags: tuple[str,...]
    parameter_role_tags: tuple[str,...]
    has_x: bool
    data_layout: str


class GenerativeModel(ABC):
    """Base for trialwise-X generators.

    No-X questionnaire generators use their own base contract because inventing
    a dummy X would obscure the scientific structure.
    """
    model_id: str
    model_version: str="v1"
    group: str
    response_family: str
    has_x: bool=True
    data_layout: str="trialwise"
    n_factors: int
    parameter_names: tuple[str,...]
    n_subjects: int
    trial_count_min: int
    trial_count_max: int
    equation: str
    description: str
    topology_tags: tuple[str,...]=()
    functional_form_tags: tuple[str,...]=()
    parameter_role_tags: tuple[str,...]=()

    @abstractmethod
    def sample_parameters(self,n_subjects:int,rng:np.random.Generator)->np.ndarray: ...
    @abstractmethod
    def sample_inputs(self,n_trials:int,rng:np.random.Generator)->np.ndarray: ...
    @abstractmethod
    def predictive_object(self,x:np.ndarray,b:np.ndarray)->np.ndarray: ...
    @abstractmethod
    def sample_observations(self,x:np.ndarray,b:np.ndarray,rng:np.random.Generator)->np.ndarray: ...
    @abstractmethod
    def log_prob(self,y:np.ndarray,x:np.ndarray,b:np.ndarray)->np.ndarray: ...

    def validate_parameters(self,b:np.ndarray)->bool:
        b=np.asarray(b)
        return b.shape[-1]==self.n_factors and bool(np.all(np.isfinite(b)))

    def validate_inputs(self,x:np.ndarray)->bool:
        return bool(np.all(np.isfinite(np.asarray(x))))

    def parameter_jacobian(self,x:np.ndarray,b:np.ndarray,eps:float=1e-5)->np.ndarray:
        x=np.asarray(x,dtype=float); b=np.asarray(b,dtype=float).reshape(-1)
        base=np.asarray(self.predictive_object(x,b),dtype=float).reshape(-1)
        jac=np.empty((base.size,self.n_factors),dtype=float)
        for j in range(self.n_factors):
            h=eps*max(1.0,abs(float(b[j])))
            bp,bm=b.copy(),b.copy(); bp[j]+=h; bm[j]-=h
            fp=np.asarray(self.predictive_object(x,bp),dtype=float).reshape(-1)
            fm=np.asarray(self.predictive_object(x,bm),dtype=float).reshape(-1)
            jac[:,j]=(fp-fm)/(2.0*h)
        return jac

    def metadata(self)->dict[str,Any]:
        return asdict(ModelMetadata(
            self.model_id,self.model_version,self.group,self.response_family,
            self.n_factors,self.parameter_names,self.n_subjects,
            self.trial_count_min,self.trial_count_max,self.equation,
            self.description,self.topology_tags,self.functional_form_tags,
            self.parameter_role_tags,bool(self.has_x),self.data_layout,
        ))

    def manifest_extras(self)->dict[str,Any]:
        return {}
