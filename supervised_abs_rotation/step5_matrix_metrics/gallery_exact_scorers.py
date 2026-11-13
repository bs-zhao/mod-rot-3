"""Exact continuous objectives for the retained frozen response-profile games.

This module intentionally does not import or read true parameter arrays.
"""
import json
from pathlib import Path

import numpy as np
import torch

import conditional_ux as conditional
import operator_mode_support as support
from shapley_function_profiles import CoalitionModels
from adaptive_coalition_models import AdaptiveCoalitionModels,ResidualAwareAdaptiveModels
from nested_polynomial_variance import NestedPolynomialVariance
from quantile_conditional_variance import QuantileConditionalVariance
from pairwise_quantile_variance import PairwiseQuantileVariance
from neighbor_conditional_variance import NeighborConditionalVariance
from predictive_variance_game import PredictiveVarianceGame

SUPPORTED={'BU','BU3','CB','CC','CD','CE','CF','CG','CH','CH2','CI','CJ','CK','BY'}


class Scorer:
    def __init__(self,bank,folder,output,tag,metric,channel,device='cuda'):
        self.bank=bank;self.data=bank.data;self.output=Path(output);self.folder=Path(folder)
        self.tag=tag;self.metric=metric;self.channel=list(self.data.channels).index(channel)
        self.device=device;self.cache={};self.models=None
        p=json.loads((self.folder/'protocol.json').read_text());self.protocol=p
        if p['source_matrix_manifest_sha256']!=bank.protocol['manifest_sha256']:raise ValueError('Matrix source mismatch')
        basis=self.folder/'shared_response_basis.npz'
        if tag=='BY':basis=self.folder.parent/'waveBU_conditional_Shapley_profiles_seed471/shared_response_basis.npz'
        with np.load(basis,allow_pickle=False) as f:
            self.y=f['latent'].copy();right=f['projection'].copy();slices=f['task_slices'];tasks=f['tasks']
        self.projections=[torch.tensor(right[:,a:b],dtype=torch.float64,device=device) for a,b in slices]
        self.tasks=tasks;self.width=max(int(b-a) for a,b in slices)
        z=self.data.u-self.data.u.mean(0);self.z=z/np.sqrt(np.var(self.data.u,axis=0).mean())
        target=torch.tensor(self.y,dtype=torch.float64,device=device);out=self.output/'conditional_models'
        if tag in ('BU','BU3','CB','CC'):
            self.models=CoalitionModels(self.z,target,out,degree=p['degree'],centers=p['radial_centers'],seed=p['seed'],device=device)
        elif tag in ('CD','CE','CF'):
            sources={r['family']:Path(r['source']) for r in p['candidate_sources']}
            factory=ResidualAwareAdaptiveModels if tag=='CE' else AdaptiveCoalitionModels
            self.models=factory(self.z,target,out,sources,device=device)
            if tag=='CF':self.game=PredictiveVarianceGame(self.models,self.projections,tasks,len(self.data.channels),self.data.dx)
        elif tag=='CG':
            self.models=NestedPolynomialVariance(self.z,target,out,bank.protocol['manifest_sha256'],device=device,
                degrees=tuple(p['degree_options']),cache_limit=128)
            if self.models.degree!=p['degree'] or not np.isclose(self.models.ridge,p['absolute_ridge'],rtol=1e-10):
                raise ArithmeticError('Restored polynomial hyperparameters changed')
        elif tag in ('CH','CH2','CI','CJ','CK'):
            factory=QuantileConditionalVariance;options=dict(target_cell_occupancy=p.get('target_proper_cell_occupancy',20))
            if tag=='CI':factory=PairwiseQuantileVariance
            if tag in ('CJ','CK'):
                factory=NeighborConditionalVariance
                options=dict(neighbors=p['neighbors'],seed=p['neighbor_seed'],
                    local_linear=p.get('local_linear_conditional_mean',False),stable_ties=p.get('stable_neighbor_tie_ordering',False))
            self.models=factory(self.z,target,out,bank.protocol['manifest_sha256'],device=device,cache_limit=128,**options)
        elif tag!='BY':raise ValueError(f'Unsupported exact gallery scorer: {tag}')

    @torch.no_grad()
    def score(self,q):
        q=np.asarray(q);key=np.round(q,10).tobytes()
        if key in self.cache:return self.cache[key]
        c=self.channel;metric=self.metric
        if self.tag=='CF':
            effects,_,_,_=self.game.frame(q)
            value=conditional.sparsity(effects[:,c,:])[metric.removeprefix('PRESS_variance_')]
        elif self.tag=='BY':
            from sklearn.tree import DecisionTreeRegressor
            import tree_haar_profiles as haar
            projections=[(int(j),p.cpu().numpy()) for (cc,j),p in zip(self.tasks,self.projections) if cc==c]
            estimator=DecisionTreeRegressor(random_state=494).fit(self.z@q,self.y)
            node,axis,_=haar.energies(estimator,self.z@q,self.y,projections,self.data.d)
            graph=node if metric.startswith('haar_node_') else axis
            value=conditional.sparsity(graph)[metric.split('_',2)[2]]
        else:
            if hasattr(self.models,'frame_spectra'):
                spectra,_=self.models.frame_spectra(q,self.projections,self.tasks,len(self.data.channels),self.data.dx)
            else:
                shares,_=self.models.frame_shares(q)
                spectra=np.zeros((len(self.data.channels),len(shares),self.data.dx,self.width))
                for (cc,j),p in zip(self.tasks,self.projections):
                    eigen=torch.linalg.svdvals(shares@p).square()/self.data.n
                    spectra[int(cc),:,int(j),:eigen.shape[1]]=eigen.cpu().numpy()
            if '_energy_' in metric:value=conditional.sparsity(spectra[c].sum(-1))[metric.split('_energy_',1)[1]]
            else:value=support.scores(spectra[c])[metric.split('_profile_',1)[1]]
        value=float(value)
        if not np.isfinite(value):raise ArithmeticError('Nonfinite exact objective')
        self.cache[key]=value
        # All subject fits remain unchanged; only discard redundant cached
        # subspace models between evaluations to keep a full batch within RAM.
        if self.models is not None and isinstance(self.models,CoalitionModels):
            for model in [self.models,*getattr(self.models,'components',{}).values()]:
                while len(model.models)>256:model.models.pop(next(iter(model.models)))
        return value
