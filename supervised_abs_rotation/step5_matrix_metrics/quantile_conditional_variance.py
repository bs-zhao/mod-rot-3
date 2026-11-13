"""Coarsened empirical conditional-variance profiles without a response form.

Proper U coalitions use coordinate-wise marginal-quantile partitions. The
full coalition conditions on exact observed U tuples. This finite asymmetric
coarsening is explicit; it is not a calibrated continuous-population effect.
"""
import hashlib
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

import run as wave


def quantile_codes(values,bins):
    """Midrank quantiles; exact boundary ties have their own symmetric class."""
    _,inverse,counts=np.unique(values,return_inverse=True,return_counts=True)
    mid=(np.cumsum(counts)-.5*counts)*bins/len(values)
    code=2*np.floor(mid).astype(int)
    boundary=np.isclose(mid,np.round(mid),rtol=0.,atol=1e-12)
    code[boundary]=2*np.round(mid[boundary]).astype(int)-1
    return code[inverse]


class QuantileConditionalVariance:
    def __init__(self,z,target,output,manifest,device='cuda',target_cell_occupancy=20,cache_limit=None):
        if len(z)<2 or target_cell_occupancy<=0:raise ValueError('Need at least two subjects and a positive occupancy')
        self.z,self.target,self.output,self.device=z,target,Path(output),device
        self.output.mkdir(parents=True,exist_ok=True);self.models=OrderedDict();self.audits=[];self.cache_limit=cache_limit
        self.bins=max(2,int(np.floor((len(z)/target_cell_occupancy)**(1/max(z.shape[1]-1,1)))))
        self.observed=target.T@target/len(z)
        _,inverse=np.unique(z,axis=0,return_inverse=True)
        self.full,self.full_audit=self.partition_covariance(inverse)
        self.unassigned=(self.observed-self.full);self.unassigned=(self.unassigned+self.unassigned.T)/2
        self.protocol_info=dict(source_matrix_manifest_sha256=manifest,degree=None,radial_centers=0,
            conditional_model='empirical conditional group means on W-only marginal-quantile partitions',
            value_function='Y.T P_S Y/N for proper coarsened input partitions; exact observed-U partition for full coalition',
            decomposition='exact conditional-covariance Shapley increments, plus fixed unresolved covariance for repeated full-U tuples',
            interpretation='finite coarsened empirical information allocation; not calibrated continuous-population Sobol or Shapley effects',
            target_proper_cell_occupancy=target_cell_occupancy,quantile_bins=self.bins,
            bins_rule='max(2,floor((N/target_occupancy)^(1/(d-1))))',
            bins_use_response_values=False,response_function_form_assumed=False,
            full_input_partition_fixed_across_rotations=True,full_input_unique_tuples=len(np.unique(inverse)),
            exact_full_vs_coarsened_proper_asymmetry=True,independence_of_inputs_assumed=False,
            full_input_prediction_generalization_claimed=False,
            numerical_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.metric_prefix='quantile_conditional_variance';self.spectrum_key='variance_eigenvalues'
        self.PSD_increments_expected=True
        wave.json_file(self.output/'full_input_partition_audit.json',self.full_audit)

    @torch.no_grad()
    def partition_covariance(self,inverse):
        counts=np.bincount(inverse);labels=torch.tensor(inverse,dtype=torch.long,device=self.device)
        mass=torch.tensor(counts,dtype=torch.float64,device=self.device)
        totals=torch.zeros((len(counts),self.target.shape[1]),dtype=torch.float64,device=self.device)
        totals.index_add_(0,labels,self.target);means=totals/mass[:,None]
        covariance=means.T@(mass[:,None]*means)/len(self.z);covariance=(covariance+covariance.T)/2
        training=self.target-means[labels];denominator=1-1/mass[labels]
        nonsingle=denominator>0
        PRESS=torch.empty_like(training)
        PRESS[nonsingle]=training[nonsingle]/denominator[nonsingle,None]
        # A singleton has no remaining within-cell observation. Use the
        # global leave-one-out mean for this diagnostic, not its own value.
        PRESS[~nonsingle]=(self.target-self.target.mean(0))[~nonsingle]*len(self.z)/(len(self.z)-1)
        return covariance,dict(groups=len(counts),minimum_occupancy=int(counts.min()),
            median_occupancy=float(np.median(counts)),maximum_occupancy=int(counts.max()),
            singleton_subject_fraction=float(np.mean(counts[inverse]==1)),
            training_relative_MSE=float(training.square().sum()/self.target.square().sum().clamp_min(1e-30)),
            fixed_partition_PRESS_relative_MSE=float(PRESS.square().sum()/self.target.square().sum().clamp_min(1e-30)),
            PRESS_scope='fixed W-only bins; proper-coalition errors include omitted-input variation; singleton uses global LOO fallback')

    @torch.no_grad()
    def proper_partition_covariance(self,inverse):return self.partition_covariance(inverse)

    @torch.no_grad()
    def covariance(self,axes):
        identifiers=[]
        for v in axes.T:
            p=np.round(np.outer(v,v),8);p[p==0.]=0.;identifiers.append(p.tobytes())
        key=tuple(sorted(identifiers))
        if key not in self.models:
            codes=np.column_stack([quantile_codes(self.z@v,self.bins) for v in axes.T])
            _,inverse=np.unique(codes,axis=0,return_inverse=True)
            covariance,audit=self.proper_partition_covariance(inverse);self.models[key]=covariance
            self.audits.append(dict(input_dimension=axes.shape[1],axis_partition_key=hashlib.sha256(b''.join(key)).hexdigest(),**audit))
            if self.cache_limit is not None:
                while len(self.models)>self.cache_limit:self.models.popitem(last=False)
        else:self.models.move_to_end(key)
        return self.models[key]

    @torch.no_grad()
    def frame_covariances(self,q):
        d=q.shape[1];values={0:torch.zeros_like(self.full),2**d-1:self.full}
        for mask in range(1,2**d-1):values[mask]=self.covariance(q[:,[i for i in range(d) if mask&(1<<i)]])
        effects=torch.zeros((d+1,*self.full.shape),dtype=torch.float64,device=self.device);smallest=0.
        for i in range(d):
            for mask in range(2**d):
                if mask&(1<<i):continue
                difference=values[mask|(1<<i)]-values[mask]
                smallest=min(smallest,float(torch.linalg.eigvalsh(difference).min()))
                effects[i]+=difference/(d*math.comb(d-1,mask.bit_count()))
        effects[-1]=self.unassigned
        error=float((effects.sum(0)-self.observed).square().sum()/self.observed.square().sum().clamp_min(1e-30))
        scale=max(float(torch.trace(self.observed)),1e-30)
        if error>1e-18:raise ArithmeticError('Quantile conditional allocation lost covariance conservation')
        if self.PSD_increments_expected and smallest < -1e-8*scale:
            raise ArithmeticError('Empirical quantile conditional allocation lost PSD nesting')
        return effects,dict(relative_squared_covariance_conservation_error=error,smallest_conditional_increment_eigenvalue=smallest)

    @torch.no_grad()
    def frame_spectra(self,q,projections,tasks,channels,dx):
        effects,audit=self.frame_covariances(q);width=max(p.shape[1] for p in projections)
        spectra=np.zeros((channels,len(effects),dx,width));negative_mass=0.
        for p,(c,j) in zip(projections,tasks):
            cmat=p.T@effects@p;cmat=(cmat+cmat.transpose(-1,-2))/2
            eigen=torch.linalg.eigvalsh(cmat).flip(-1);negative_mass+=float((-eigen.clamp_max(0)).sum())
            spectra[c,:,j,:eigen.shape[1]]=eigen.clamp_min(0).cpu().numpy()
        audit.update(clipped_roundoff_negative_mass=negative_mass)
        return spectra,audit

    def finalize(self):return dict(conditional_partitions=len(self.models),quantile_bins=self.bins)
