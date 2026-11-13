"""Exact label-free scorer of the frozen CF predictive-variance UX graph."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import conditional_ux as conditional
import matrix_function_profiles as profiles
import run as wave
from adaptive_coalition_models import AdaptiveCoalitionModels
from predictive_variance_game import PredictiveVarianceGame

KEY='UX_columns_worst_soft_count_eps003'


class Scorer:
    def __init__(self,source,original,output,device='cuda',bank_folder=None):
        original,output=Path(original),Path(output)
        protocol=json.loads((original/'protocol.json').read_text())
        if protocol['status']!='COMPLETE' or hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
            raise ValueError('Frozen predictive variance scores changed')
        self.bank=profiles.MatrixBank(source,bank_folder or original.parent/'matrix_response_table_seed383');self.data=self.bank.data
        if self.bank.protocol['manifest_sha256']!=protocol['source_matrix_manifest_sha256']:raise ValueError('Actual source matrices changed')
        with np.load(original/'shared_response_basis.npz',allow_pickle=False) as saved:
            y=saved['latent'].copy();projections=saved['projection'].copy();slices=saved['task_slices'];tasks=saved['tasks']
        p=[torch.tensor(projections[:,a:b],dtype=torch.float64,device=device) for a,b in slices]
        z=self.data.u-self.data.u.mean(0);z/=np.sqrt(np.var(self.data.u,axis=0).mean())
        sources={entry['family']:Path(entry['source']) for entry in protocol['candidate_sources']}
        self.models=AdaptiveCoalitionModels(z,torch.tensor(y,dtype=torch.float64,device=device),output/'new_conditional_models',sources,device=device)
        self.game=PredictiveVarianceGame(self.models,p,tasks,len(self.data.channels),self.data.dx)
        self.channel=list(self.data.channels).index('logit_nn');self.maximum_efficiency_error=0.;self.initial_models=len(self.models.models);self.cache={}

    @torch.no_grad()
    def score(self,q,force=False):
        keys=[]
        for v in np.asarray(q).T:
            projector=np.round(np.outer(v,v),8);projector[projector==0.]=0.;keys.append(projector.tobytes())
        key=b''.join(sorted(keys))
        if not force and key in self.cache:return self.cache[key]
        effects,_,_,audit=self.game.frame(q)
        self.maximum_efficiency_error=max(self.maximum_efficiency_error,audit['maximum_conservation_error']**2)
        value=conditional.sparsity(effects[:,self.channel,:])[KEY];self.cache[key]=value
        return value

    def finalize(self):
        wave.json_file(self.models.output.parent/'fixed_fit_PRESS_audits.json',self.game.audits)
        return self.models.finalize()
