"""W-only hierarchical basis innovations, preserving the complete fit space.

All groups of the same interaction order use the same lower-order span.
No ordering of individual U axes or outcome information defines projections.
This is an empirical hierarchy, not independent-input functional ANOVA.
"""
import numpy as np
import torch

import group_sparse_gp as base


def dictionary(z,degree,radial,reference=None,device='cuda'):
    raw,groups,masks,original=base.dictionary(z,degree,radial)
    x=torch.tensor(raw,dtype=torch.float64,device=device);result=x.clone()
    group_order=np.array([int(mask).bit_count() for mask in masks])
    column_order=group_order[groups];gram=x.T@x;audits=[]
    for order in np.unique(column_order):
        lower=np.flatnonzero(column_order<order);current=np.flatnonzero(column_order==order)
        if not len(lower):continue
        g=gram[lower][:,lower];values,vectors=torch.linalg.eigh((g+g.T)/2)
        good=values>max(float(values.max())*1e-10,1e-12)
        rhs=gram[lower][:,current]
        coefficient=(vectors[:,good]/values[good])@(vectors[:,good].T@rhs)
        result[:,current]-=x[:,lower]@coefficient
        cross=x[:,lower].T@result[:,current]
        bound=torch.linalg.matrix_norm(cross)/(torch.linalg.matrix_norm(x[:,lower])*torch.linalg.matrix_norm(result[:,current])).clamp_min(1e-30)
        audits.append(dict(interaction_order=int(order),lower_columns=len(lower),retained_lower_rank=int(good.sum()),
            normalized_lower_span_cross_covariance=float(bound),projection_relative_eigen_cutoff=1e-10))
    fit=result if reference is None else result[reference]
    mean=fit.mean(0);centered_fit=fit-mean;result=result-mean;scales=[];fractions=[]
    for group in range(len(masks)):
        which=groups==group
        scale=max(float(centered_fit[:,which].square().sum()/len(fit))**.5,1e-30)
        fractions.append(float(result[:,which].square().sum()/x[:,which].square().sum().clamp_min(1e-30)))
        result[:,which]/=scale;scales.append(scale)
    audit=dict(degree=degree,radial_features=radial.shape[1],features=result.shape[1],group_masks=masks.tolist(),
        group_scales=scales,feature_mean=mean.cpu().numpy().tolist(),original_dictionary=original,
        hierarchy=audits,group_innovation_energy_fraction=fractions,
        projection_inputs='W only, all subjects',
        full_space_preserved='subtract lower-order feature combinations only; invertible block-triangular basis change before scaling',
        normalization='unit reference-subject feature trace per U-subset after common lower-order projection')
    return result.cpu().numpy(),groups,masks,audit
