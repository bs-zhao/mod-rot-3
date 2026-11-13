"""Data-only kNN conditional residual covariance and signed Shapley operators.

Adapt the observed-data conditional variance estimator (18) of Broto,
Bachoc and Depecker (2020) to matrix-valued response functions. No hybrid
inputs or new response evaluations are used.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from adaptive_coalition_models import subspace_key
from pairwise_quantile_variance import PairwiseQuantileVariance


class NeighborConditionalVariance(PairwiseQuantileVariance):
    def __init__(self,*args,neighbors=3,seed=471,local_linear=False,stable_ties=False,**kwargs):
        super().__init__(*args,**kwargs)
        if not 2<=neighbors<=len(self.z):raise ValueError('Need between two and N neighbors')
        self.neighbors,self.seed=neighbors,seed
        self.local_linear,self.stable_ties=local_linear,stable_ties
        if local_linear and neighbors<=self.z.shape[1]:raise ValueError('Local affine covariance needs more neighbors than U dimensions')
        self.metric_prefix='knn_conditional_cross_covariance';self.bins=None
        if local_linear:self.metric_prefix='knn_local_linear_cross_covariance'
        for key in ('target_proper_cell_occupancy','quantile_bins','bins_rule','bins_use_response_values',
                    'singleton_proper_cells','exact_full_vs_coarsened_proper_asymmetry',
                    'output_basis_and_bin_selection_not_independent_of_sample'):
            self.protocol_info.pop(key,None)
        self.protocol_info.update(conditional_model='all-anchor kNN sample covariance of existing observed response profiles',
            value_function='observed response covariance minus mean within-neighborhood sample covariance for proper input coalitions',
            decomposition='exact all-coalition Shapley aggregation of finite kNN covariance estimates',
            interpretation='signed data-only kNN conditional covariance diagnostic, not a finite-sample PSD or population-identification guarantee',
            neighbors=neighbors,neighbor_seed=seed,all_subjects_are_anchors=True,
            local_linear_conditional_mean=local_linear,stable_neighbor_tie_ordering=stable_ties,
            local_mean_degree=1 if local_linear else 0,
            neighbor_metric='Euclidean on a U subspace with one shared global U scale',
            hybrid_input_evaluations_used=False,new_response_evaluations_used=False,
            output_basis_derived_from_same_response_sample=True,
            conditional_variance_reference='https://arxiv.org/html/1812.09168v2#S6.SS1.SSS1',
            neighbor_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if local_linear:self.protocol_info.update(
            conditional_model='all-anchor kNN local affine residual covariance with residual degrees-of-freedom correction',
            local_mean_interpretation='local first-order numerical approximation; no prescribed global response function',
            conditional_variance_reference='own local-affine extension of the data-only kNN variance estimator')

    def neighbor_indices(self,basis,key):
        points=self.z@basis;tree=cKDTree(points);k=min(self.neighbors+1,len(points))
        distance,indices=tree.query(points,k=k)
        ties=np.zeros(len(points),dtype=bool)
        if k>self.neighbors:
            ties=np.isclose(distance[:,self.neighbors-1],distance[:,self.neighbors],rtol=1e-12,atol=1e-12)
            for anchor in np.flatnonzero(ties):
                radius=distance[anchor,self.neighbors-1]
                candidates=np.asarray(tree.query_ball_point(points[anchor],radius*(1+1e-12)+1e-12))
                if self.stable_ties:candidates=np.sort(candidates)
                ds=np.linalg.norm(points[candidates]-points[anchor],axis=1)
                rng=np.random.default_rng(self.seed+int(key[1][:8],16)+int(anchor))
                order=np.lexsort((rng.random(len(candidates)),np.round(ds,12)))
                indices[anchor,:self.neighbors]=candidates[order[:self.neighbors]]
        return indices[:,:self.neighbors].copy(),dict(boundary_distance_tie_anchor_fraction=float(ties.mean()),
            neighbor_radius_quantiles=np.quantile(distance[:,self.neighbors-1],[0,.5,.9,.99,1]).tolist())

    @torch.no_grad()
    def covariance(self,axes):
        basis=np.linalg.qr(axes)[0][:,:axes.shape[1]];key=subspace_key(basis)
        if key not in self.models:
            indices,audit=self.neighbor_indices(basis,key)
            tensor=torch.tensor(indices,dtype=torch.long,device=self.device)
            if self.local_linear:
                points=self.z@basis;coordinates=points[indices]-points[:,None,:]
                design=torch.tensor(np.concatenate((np.ones((*indices.shape,1)),coordinates),axis=2),dtype=torch.float64,device=self.device)
                left,singular,_=torch.linalg.svd(design,full_matrices=False)
                good=singular>singular.max(1,keepdim=True).values*1e-10
                freedom=self.neighbors-good.sum(1)
                if torch.any(freedom<=0):raise ArithmeticError('No local residual degrees of freedom')
                weights=1/freedom.to(torch.float64)
                projected=torch.einsum('nkp,nkr->npr',left,self.target[tensor])
                projected*=good[:,:,None]*torch.sqrt(weights)[:,None,None]
                flat=projected.reshape(-1,self.target.shape[1])
                counts=torch.zeros(len(self.z),dtype=torch.float64,device=self.device)
                counts.scatter_add_(0,tensor.reshape(-1),weights[:,None].expand_as(tensor).reshape(-1))
                residual=(self.target.T@(counts[:,None]*self.target)-flat.T@flat)/len(self.z)
                audit.update(local_residual_degrees_of_freedom_min=int(freedom.min()),
                    local_residual_degrees_of_freedom_max=int(freedom.max()),local_mean_degree=1)
            else:
                means=self.target[tensor].mean(1)
                counts=torch.tensor(np.bincount(indices.ravel(),minlength=len(self.z)),dtype=torch.float64,device=self.device)
                residual=(self.target.T@(counts[:,None]*self.target)-self.neighbors*(means.T@means))/(len(self.z)*(self.neighbors-1))
            residual=(residual+residual.T)/2
            minimum=float(torch.linalg.eigvalsh(residual).min())
            if minimum < -1e-8*max(float(torch.trace(self.observed)),1e-30):
                raise ArithmeticError('Within-neighbor residual covariance lost PSD beyond roundoff')
            self.models[key]=self.observed-residual
            self.audits.append(dict(input_dimension=key[0],projector_key=key[1],neighbors=self.neighbors,
                all_anchor_subjects=len(self.z),smallest_residual_covariance_eigenvalue=minimum,
                conditional_residual_trace_relative_to_observed=float(torch.trace(residual)/torch.trace(self.observed).clamp_min(1e-30)),**audit))
            if self.cache_limit is not None:
                while len(self.models)>self.cache_limit:self.models.popitem(last=False)
        else:self.models.move_to_end(key)
        return self.models[key]

    def finalize(self):return dict(cached_conditional_neighbor_subspaces=len(self.models),conditional_neighbor_evaluations=len(self.audits),neighbors=self.neighbors)
