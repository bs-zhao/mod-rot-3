"""Label-free exact scorer for the frozen CG polynomial covariance criterion."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import matrix_function_profiles as profiles
import operator_mode_support as mode_support
from nested_polynomial_variance import NestedPolynomialVariance

KEY='UX_operator_columns_mean_soft_count_eps003'


class Scorer:
    def __init__(self,source,original,output,device='cuda',bank_folder=None):
        original,output=Path(original),Path(output);self.original=original;self.output=output;self.device=device
        protocol=json.loads((original/'protocol.json').read_text())
        if protocol['status']!='COMPLETE' or hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
            raise ValueError('Frozen covariance source changed')
        self.bank=profiles.MatrixBank(source,bank_folder or original.parent/'matrix_response_table_seed383');self.data=self.bank.data
        if protocol['source_matrix_manifest_sha256']!=self.bank.protocol['manifest_sha256']:raise ValueError('Actual matrices changed')
        with np.load(original/'shared_response_basis.npz',allow_pickle=False) as saved:
            y=saved['latent'].copy();projection=saved['projection'].copy();slices=saved['task_slices'];tasks=saved['tasks']
        self.projections=[];self.tasks=[];channel=list(self.data.channels).index('p_nn')
        for (c,j),(a,b) in zip(tasks,slices):
            if c==channel:
                self.projections.append(torch.tensor(projection[:,a:b],dtype=torch.float64,device=device));self.tasks.append((0,int(j)))
        z=self.data.u-self.data.u.mean(0);z/=np.sqrt(np.var(self.data.u,axis=0).mean())
        self.models=NestedPolynomialVariance(z,torch.tensor(y,dtype=torch.float64,device=device),output/'new_covariance_models',
            self.bank.protocol['manifest_sha256'],device=device,degrees=tuple(protocol['degree_options']),cache_limit=512)
        if self.models.degree!=protocol['degree'] or not np.isclose(self.models.ridge,protocol['absolute_ridge'],rtol=1e-10):
            raise ArithmeticError('Restored covariance hyperparameters changed')
        with np.load(original/'conditional_models'/'full_input_model.npz',allow_pickle=False) as saved:
            old=saved['coefficient'].copy()
        with np.load(output/'new_covariance_models'/'full_input_model.npz',allow_pickle=False) as saved:
            current=saved['coefficient'].copy()
        if np.sum((old-current)**2)>1e-16*max(np.sum(old**2),1e-30):
            raise ArithmeticError('Restored full-input coefficients changed')
        self.initial_models=0;self.maximum_efficiency_error=0.;self.cache={}

    @torch.no_grad()
    def score(self,q,force=False):
        keys=[]
        for v in np.asarray(q).T:
            projector=np.round(np.outer(v,v),8);projector[projector==0.]=0.;keys.append(projector.tobytes())
        key=b''.join(sorted(keys))
        if not force and key in self.cache:return self.cache[key]
        spectra,audit=self.models.frame_spectra(q,self.projections,self.tasks,1,self.data.dx)
        self.maximum_efficiency_error=max(self.maximum_efficiency_error,audit['relative_squared_covariance_conservation_error'])
        value=mode_support.scores(spectra[0])[KEY];self.cache[key]=value
        return value

    def finalize(self):
        return self.models.finalize()
