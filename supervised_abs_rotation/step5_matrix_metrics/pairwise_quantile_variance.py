"""Off-diagonal within-bin pair correction; signed spectra are kept explicit."""
import hashlib
from pathlib import Path

import numpy as np
import torch

from quantile_conditional_variance import QuantileConditionalVariance


class PairwiseQuantileVariance(QuantileConditionalVariance):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.PSD_increments_expected=False
        self.metric_prefix='pairwise_quantile_cross_covariance'
        self.spectrum_key='absolute_eigenvalues_of_signed_components'
        self.protocol_info.update(conditional_model='within-quantile-bin ordered off-diagonal pair cross products',
            value_function='sum_g (s_g s_g.T - sum_in_g y_i y_i.T)/(N*(n_g-1)) for proper coalitions with n_g>1; exact observed-U full endpoint',
            interpretation='signed finite within-bin pair correction, not PSD or calibrated population Sobol effects',
            finite_sample_PSD_guarantee=False,score_uses_absolute_eigenvalues=True,
            singleton_proper_cells='no available pair; zero estimated conditional cross-product contribution, coverage audited',
            output_basis_and_bin_selection_not_independent_of_sample=True,
            pairwise_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    @torch.no_grad()
    def proper_partition_covariance(self,inverse):
        _,audit=self.partition_covariance(inverse)
        counts=np.bincount(inverse);labels=torch.tensor(inverse,dtype=torch.long,device=self.device)
        mass=torch.tensor(counts,dtype=torch.float64,device=self.device)
        totals=torch.zeros((len(counts),self.target.shape[1]),dtype=torch.float64,device=self.device)
        totals.index_add_(0,labels,self.target)
        weight=torch.where(mass>1,1/(mass-1).clamp_min(1.),torch.zeros_like(mass))
        covariance=(totals.T@(weight[:,None]*totals)-self.target.T@(weight[labels,None]*self.target))/len(self.z)
        covariance=(covariance+covariance.T)/2
        audit.update(within_bin_self_pairs_removed=True,subject_fraction_with_available_pairs=float(np.mean(counts[inverse]>1)),
            smallest_corrected_covariance_eigenvalue=float(torch.linalg.eigvalsh(covariance).min()))
        return covariance,audit

    @torch.no_grad()
    def frame_spectra(self,q,projections,tasks,channels,dx):
        effects,audit=self.frame_covariances(q);width=max(p.shape[1] for p in projections)
        spectra=np.zeros((channels,len(effects),dx,width));negative=0.;absolute=0.
        for p,(c,j) in zip(projections,tasks):
            covariance=p.T@effects@p;covariance=(covariance+covariance.transpose(-1,-2))/2
            eigen=torch.linalg.eigvalsh(covariance)
            negative+=float((-eigen.clamp_max(0)).sum());absolute+=float(eigen.abs().sum())
            spectra[c,:,j,:eigen.shape[1]]=torch.sort(eigen.abs(),descending=True).values.cpu().numpy()
        audit.update(negative_projected_spectral_mass=negative,absolute_projected_spectral_mass=absolute,
            negative_fraction_of_absolute_spectral_mass=negative/max(absolute,1e-30),
            absolute_spectra_are_not_a_PSD_covariance_decomposition=True)
        return spectra,audit
