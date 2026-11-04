from __future__ import annotations

from abc import abstractmethod
import numpy as np

from ..base import GenerativeModel


def _clip_probability(p, eps=1e-12):
    return np.clip(np.asarray(p,dtype=float),eps,1.0-eps)


class ChoiceModel(GenerativeModel):
    group="choice"
    response_family="discrete_choice"
    has_x=True
    data_layout="trialwise"


class BinaryChoiceModel(ChoiceModel):
    """Base contract for Bernoulli choice generators.

    The predictive object is P(Y=1 | X,b).  Concrete models only need to define
    the probability surface and parameter/input distributions.
    """

    num_choices=2

    @abstractmethod
    def choice_probability(self,x:np.ndarray,b:np.ndarray)->np.ndarray: ...

    def predictive_object(self,x,b):
        return np.asarray(self.choice_probability(x,b),dtype=float)

    def sample_observations(self,x,b,rng):
        p=_clip_probability(self.choice_probability(x,b))
        return rng.binomial(1,p,size=p.shape).astype(np.int64)

    def log_prob(self,y,x,b):
        y=np.asarray(y,dtype=float)
        p=_clip_probability(self.choice_probability(x,b))
        return y*np.log(p)+(1.0-y)*np.log1p(-p)

    def fisher_information(self,x,b,*,average=True,eps=1e-12):
        """Bernoulli Fisher information in the model parameterization.

        If J = d p(X,b) / d b, then each trial contributes
        J J^T / [p(1-p)].  This stays generic even when the concrete generator
        is not a linear-logit model.
        """
        x=np.asarray(x,dtype=float)
        p=_clip_probability(self.choice_probability(x,b),eps)
        j=np.asarray(self.parameter_jacobian(x,b),dtype=float)
        w=1.0/np.maximum(p*(1.0-p),eps)
        info=j.T@(w[:,None]*j)
        return info/max(len(x),1) if average else info

    def theoretical_prediction_ceiling(self):
        # There is no R2-style ceiling for Bernoulli observations.  The
        # precheck reports Bayes accuracy, proper-score performance, and null
        # improvement instead.
        return None
