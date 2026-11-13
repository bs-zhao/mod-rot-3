"""Label-free logit conditional-variance mode support from the frozen CH2 game."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import matrix_function_profiles as profiles
import operator_mode_support as mode_support
from quantile_conditional_variance import QuantileConditionalVariance

KEY='UX_operator_columns_mean_soft_count_eps010'


class Scorer:
    def __init__(self,source,original,output,device='cuda',bank_folder=None,model_factory=QuantileConditionalVariance,model_options=None,
                 channel='logit_nn',metric_key=KEY):
        original,output=Path(original),Path(output)
        protocol=json.loads((original/'protocol.json').read_text())
        if protocol['status']!='COMPLETE' or hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
            raise ValueError('Frozen quantile scores changed')
        self.bank=profiles.MatrixBank(source,bank_folder or original.parent/'matrix_response_table_seed383');self.data=self.bank.data
        if self.bank.protocol['manifest_sha256']!=protocol['source_matrix_manifest_sha256']:raise ValueError('Actual matrices changed')
        with np.load(original/'shared_response_basis.npz',allow_pickle=False) as saved:
            y=saved['latent'].copy();projection=saved['projection'].copy();slices=saved['task_slices'];tasks=saved['tasks']
        self.projections=[];self.tasks=[];channel_index=list(self.data.channels).index(channel);self.metric_key=metric_key
        for (c,j),(a,b) in zip(tasks,slices):
            if c==channel_index:
                self.projections.append(torch.tensor(projection[:,a:b],dtype=torch.float64,device=device));self.tasks.append((0,int(j)))
        z=self.data.u-self.data.u.mean(0);z/=np.sqrt(np.var(self.data.u,axis=0).mean())
        options=model_options if model_options is not None else dict(target_cell_occupancy=protocol['target_proper_cell_occupancy'])
        self.models=model_factory(z,torch.tensor(y,dtype=torch.float64,device=device),output/'new_conditional_partitions',
            self.bank.protocol['manifest_sha256'],device=device,cache_limit=512,**options)
        if protocol.get('quantile_bins') is not None and self.models.bins!=protocol['quantile_bins']:
            raise ArithmeticError('Restored bin resolution changed')
        self.initial_models=0;self.maximum_efficiency_error=0.;self.cache={}

    @torch.no_grad()
    def score(self,q,force=False):
        keys=[]
        for v in np.asarray(q).T:
            p=np.round(np.outer(v,v),8);p[p==0.]=0.;keys.append(p.tobytes())
        key=b''.join(sorted(keys))
        if not force and key in self.cache:return self.cache[key]
        spectra,audit=self.models.frame_spectra(q,self.projections,self.tasks,1,self.data.dx)
        self.maximum_efficiency_error=max(self.maximum_efficiency_error,audit['relative_squared_covariance_conservation_error'])
        value=mode_support.scores(spectra[0])[self.metric_key];self.cache[key]=value
        return value

    def finalize(self):return self.models.finalize()
