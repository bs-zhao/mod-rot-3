"""C1: balanced binary-choice analogue of continuous P1."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from .base import BinaryChoiceModel


class ChoiceC1LinearLogit(BinaryChoiceModel):
    """Four-factor binary-choice generator with a linear logit.

    b1 controls overall choice tendency.  b2-b4 independently modulate the
    effects of X1-X3.  The sqrt(3) scaling gives each design feature unit
    variance under X_j ~ Uniform(-1,1), matching the balancing logic of P1.
    """

    model_id="C1"
    model_version="v1"
    n_factors=4
    parameter_names=("b1","b2","b3","b4")
    n_subjects=5000
    trial_count_min=300
    trial_count_max=500
    n_input_features=3

    parameter_sd=0.5

    equation=(
        "eta=b1+sqrt(3)b2*x1+sqrt(3)b3*x2+sqrt(3)b4*x3; "
        "p=sigmoid(eta); y~Bernoulli(p)"
    )
    description=(
        "Balanced binary-choice benchmark with one overall choice-tendency "
        "factor and three separate stimulus-slope factors."
    )
    topology_tags=("global","one-b-one-x")
    functional_form_tags=("linear-logit","sigmoid")
    parameter_role_tags=("choice-bias","slope")

    def sample_parameters(self,n_subjects,rng):
        return rng.normal(
            0.0,self.parameter_sd,size=(int(n_subjects),self.n_factors)
        )

    def sample_inputs(self,n_trials,rng):
        return rng.uniform(
            -1.0,1.0,size=(int(n_trials),self.n_input_features)
        )

    @staticmethod
    def design_matrix(x):
        x=np.asarray(x,dtype=float)
        if x.ndim!=2 or x.shape[1]!=3:
            raise ValueError("C1 expects x with shape (n_trials,3).")
        return np.column_stack(
            [np.ones(x.shape[0],dtype=float),np.sqrt(3.0)*x]
        )

    def linear_predictor(self,x,b):
        b=np.asarray(b,dtype=float)
        if b.shape[-1]!=4:
            raise ValueError("C1 expects b[...,4].")
        design=self.design_matrix(x)
        if b.ndim==1:
            return design@b
        return np.einsum("nd,...d->...n",design,b)

    def choice_probability(self,x,b):
        return expit(self.linear_predictor(x,b))

    def parameter_jacobian(self,x,b,eps=1e-5):
        del eps
        design=self.design_matrix(x)
        p=np.asarray(self.choice_probability(x,b),dtype=float).reshape(-1)
        return (p*(1.0-p))[:,None]*design

    def fisher_information(self,x,b,*,average=True,eps=1e-12):
        del eps
        design=self.design_matrix(x)
        p=np.asarray(self.choice_probability(x,b),dtype=float).reshape(-1)
        w=p*(1.0-p)
        info=design.T@(w[:,None]*design)
        return info/max(len(design),1) if average else info

    def oracle_fit(self,x,y,initial=None,max_iter=50,tol=1e-9):
        """Fast unpenalized logistic MLE via damped Newton/IRLS."""
        design=self.design_matrix(x)
        y=np.asarray(y,dtype=float).reshape(-1)
        if y.shape[0]!=design.shape[0]:
            raise ValueError("x and y must contain the same number of trials.")
        if not np.all((y==0)|(y==1)):
            raise ValueError("C1 oracle fit requires binary responses.")

        beta=(
            np.zeros(self.n_factors,dtype=float)
            if initial is None
            else np.asarray(initial,dtype=float).reshape(self.n_factors).copy()
        )
        eye=np.eye(self.n_factors)
        previous=np.inf
        for _ in range(int(max_iter)):
            eta=design@beta
            p=expit(eta)
            # Bernoulli negative log likelihood in a stable form.
            objective=float(np.sum(np.logaddexp(0.0,eta)-y*eta))
            gradient=design.T@(p-y)
            w=np.maximum(p*(1.0-p),1e-8)
            hessian=design.T@(w[:,None]*design)+1e-8*eye
            try:
                step=np.linalg.solve(hessian,gradient)
            except np.linalg.LinAlgError:
                step=np.linalg.lstsq(hessian,gradient,rcond=None)[0]

            # Backtracking prevents rare Newton overshoots under near separation.
            scale=1.0
            accepted=False
            for _ in range(20):
                proposal=beta-scale*step
                eta_new=design@proposal
                new_objective=float(
                    np.sum(np.logaddexp(0.0,eta_new)-y*eta_new)
                )
                if new_objective<=objective+1e-12:
                    beta=proposal
                    previous=new_objective
                    accepted=True
                    break
                scale*=0.5
            if not accepted:
                break
            if np.linalg.norm(scale*step)<=tol*(1.0+np.linalg.norm(beta)):
                break
            if abs(objective-previous)<=tol*(1.0+abs(objective)):
                break
        return beta

    def manifest_extras(self):
        return {
            "parameter_prior":"b1..b4 independent Normal(0,0.5^2)",
            "input_distribution":"x1..x3 iid Uniform(-1,1)",
            "observation_distribution":"Bernoulli(sigmoid(eta))",
            "parameter_sd":self.parameter_sd,
            "num_choices":self.num_choices,
            "design_feature_second_moment":[1.0,1.0,1.0,1.0],
            "balance_rationale":(
                "The intercept and sqrt(3)-scaled slope features all have "
                "E[phi_j^2]=1; symmetric zero-centered priors/design yield "
                "population choice rate 0.5 and near-equal Fisher information."
            ),
        }
