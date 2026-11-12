"""Label-free scorer reusing frozen observed profiles and conditional models."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import matrix_function_profiles as profiles
import operator_mode_support as mode_support
import shapley_function_profiles as shapley

KEY='UX_operator_columns_mean_soft_count_eps003'


class Scorer:
    def __init__(self,source,original,output,device='cuda',bank_folder=None):
        self.original,self.output,self.device=Path(original),Path(output),device
        self.protocol=json.loads((self.original/'protocol.json').read_text())
        digest=hashlib.sha256((self.original/'blind_scores.csv').read_bytes()).hexdigest()
        if self.protocol['status']!='COMPLETE' or digest!=self.protocol['scores_sha256']:
            raise ValueError('Source Shapley profile scores changed')
        if self.protocol['response_transform']!='identity':raise ValueError('This scorer currently expects the raw-profile source')
        parent=self.original.parent
        self.bank=profiles.MatrixBank(source,bank_folder or parent/'matrix_response_table_seed383');self.data=self.bank.data
        if self.bank.protocol['manifest_sha256']!=self.protocol['source_matrix_manifest_sha256']:
            raise ValueError('Actual matrix profile source changed')
        with np.load(self.original/'shared_response_basis.npz',allow_pickle=False) as saved:
            latent=saved['latent'].copy();projection=saved['projection'].copy();slices=saved['task_slices'];tasks=saved['tasks']
        c=list(self.data.channels).index('p_nn');self.projections=[];self.width=max(int(b-a) for a,b in slices)
        for (cc,j),(a,b) in zip(tasks,slices):
            if cc==c:self.projections.append((int(j),torch.tensor(projection[:,a:b],dtype=torch.float64,device=device)))
        z=self.data.u-self.data.u.mean(0);z/=np.sqrt(np.var(self.data.u,axis=0).mean())
        self.models=shapley.CoalitionModels(z,torch.tensor(latent,dtype=torch.float64,device=device),self.output/'new_conditional_models',
            degree=self.protocol['degree'],centers=self.protocol['radial_centers'],device=device)
        for file in sorted((self.original/'conditional_models').glob('model*.npz')):
            with np.load(file,allow_pickle=False) as saved:
                basis=saved['basis'].copy();coefficient=saved['coefficient'].copy();length=float(saved['lengthscale'])
            if coefficient.shape[1]!=latent.shape[1]:raise ValueError('A coalition coefficient uses a different response basis')
            projector=basis@basis.T;canonical=np.round(projector,6);canonical[canonical==0.]=0.
            key=(basis.shape[1],hashlib.sha256(canonical.tobytes()).hexdigest())
            if key in self.models.models and np.max(np.abs(self.models.models[key]['basis']@self.models.models[key]['basis'].T-projector))>1e-6:
                key=(basis.shape[1],hashlib.sha256(projector.tobytes()).hexdigest())
            self.models.models[key]=dict(basis=basis,coefficient=coefficient,lengthscale=length)
        self.initial_models=len(self.models.models);self.cache={};self.maximum_efficiency_error=0.

    @torch.no_grad()
    def score(self,q,force=False):
        keys=[]
        for v in np.asarray(q).T:
            projector=np.round(np.outer(v,v),8);projector[projector==0.]=0.;keys.append(projector.tobytes())
        key=b''.join(sorted(keys))
        if not force and key in self.cache:return self.cache[key]
        shares,error=self.models.frame_shares(q);self.maximum_efficiency_error=max(self.maximum_efficiency_error,error)
        spectra=np.zeros((self.data.d,self.data.dx,self.width))
        for j,projection in self.projections:
            component=shares@projection
            eigen=torch.linalg.svdvals(component).square()/self.data.n
            spectra[:,j,:eigen.shape[1]]=eigen.cpu().numpy()
        value=mode_support.scores(spectra)[KEY];self.cache[key]=value
        return value
