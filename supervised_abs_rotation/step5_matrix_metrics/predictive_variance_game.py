"""Reusable all-subject fixed-fit PRESS predictive game; no angle labels."""
import numpy as np
import torch

from adaptive_coalition_models import subspace_key
from predictive_variance_effects import allocate


class PredictiveVarianceGame:
    def __init__(self,models,projections,tasks,channels,dx):
        self.models,self.projections,self.tasks=models,projections,tasks
        self.shape=(channels,dx);self.risks={};self.audits=[]
        y=models.target;n=len(y)
        self.empty=self.task_risks((y-y.mean(0))*n/(n-1));self.active=self.empty>1e-30
        self.full=self.risk(np.eye(models.z.shape[1]))

    @torch.no_grad()
    def task_risks(self,residual):
        result=np.zeros(self.shape)
        for p,(c,j) in zip(self.projections,self.tasks):
            result[c,j]=float((residual@p).square().mean(0).sum())
        return result

    @torch.no_grad()
    def risk(self,axes):
        key=subspace_key(np.linalg.qr(axes)[0][:,:axes.shape[1]])
        if key not in self.risks:
            PRESS,numerical=self.models.fixed_fit_PRESS(axes);self.risks[key]=self.task_risks(PRESS)
            self.audits.append(dict(input_dimension=key[0],projector_key=key[1],PRESS_MSE=self.risks[key].tolist(),**numerical))
        return self.risks[key]

    @torch.no_grad()
    def frame(self,q):
        d=q.shape[1];raw=np.zeros((2**d,*self.shape))
        for mask in range(1,2**d):
            selected=[i for i in range(d) if mask&(1<<i)]
            error=self.full if mask==2**d-1 else self.risk(q[:,selected])
            raw[mask]=np.divide(self.empty-error,self.empty,out=np.zeros(self.shape),where=self.active)
        effects,game,audit=allocate(raw,d);effects*=self.active[None]
        return effects,raw,game,audit
