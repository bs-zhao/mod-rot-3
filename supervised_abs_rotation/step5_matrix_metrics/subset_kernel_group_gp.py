"""Hierarchical all-subset GP dictionary, with polynomial approximation support.

Every nonempty U subset receives the same eight-center W-only radial policy.
All groups at a given order share the complete lower-order projection span.
No response equation, task label or preferred axis enters this construction.
"""
import numpy as np
import torch

import run as wave
import conditional_ui_witness as ui


def dictionary(z,degree,radial,reference=None,device='cuda'):
    d=z.shape[1];centers=8
    powers=wave.sur.powers_for(d,degree)[1:]
    poly=wave.sur.monomials(z,powers,wave.sur.weights_for(powers))
    poly_masks=np.sum((powers>0)*(2**np.arange(d))[None],axis=1)
    blocks,labels,kernel_audits=[],[],[]
    for mask in range(1,2**d):
        columns=[i for i in range(d) if mask&(1<<i)]
        # The same seed chooses the same subject IDs in every subset.
        # Axis permutations and sign flips therefore preserve this dictionary.
        features,audit=ui.nystrom(z[:,columns],centers,448)
        features-=features.mean(0)
        block=poly[:,poly_masks==mask]
        block-=block.mean(0)
        parts=[]
        for part in (block,features):
            if part.shape[1]:parts.append(part/max(float(np.mean(np.sum(part*part,axis=1)))**.5,1e-30))
        combined=np.column_stack(parts)/np.sqrt(len(parts))
        blocks.append(combined);labels.extend([mask]*combined.shape[1]);kernel_audits.append(dict(mask=mask,**audit))
    raw=np.column_stack(blocks);masks=np.arange(1,2**d);groups=np.array(labels)-1
    x=torch.tensor(raw,dtype=torch.float64,device=device);result=x.clone();gram=x.T@x
    order=np.array([int(mask).bit_count() for mask in labels]);history=[]
    for level in np.unique(order):
        lower=np.flatnonzero(order<level);current=np.flatnonzero(order==level)
        if not len(lower):continue
        g=gram[lower][:,lower];values,vectors=torch.linalg.eigh((g+g.T)/2)
        good=values>max(float(values.max())*1e-10,1e-12)
        coefficient=(vectors[:,good]/values[good])@(vectors[:,good].T@gram[lower][:,current])
        result[:,current]-=x[:,lower]@coefficient
        cross=x[:,lower].T@result[:,current]
        history.append(dict(interaction_order=int(level),lower_columns=len(lower),retained_lower_rank=int(good.sum()),
            normalized_lower_span_cross_covariance=float(torch.linalg.matrix_norm(cross)/
                (torch.linalg.matrix_norm(x[:,lower])*torch.linalg.matrix_norm(result[:,current])).clamp_min(1e-30))))
    fit=result if reference is None else result[reference];mean=fit.mean(0)
    centered_fit=fit-mean;result=result-mean;scales=[];fractions=[]
    for group in range(len(masks)):
        which=groups==group
        scale=max(float(centered_fit[:,which].square().sum()/len(fit))**.5,1e-30)
        fractions.append(float(result[:,which].square().sum()/x[:,which].square().sum().clamp_min(1e-30)))
        result[:,which]/=scale;scales.append(scale)
    return result.cpu().numpy(),groups,masks,dict(degree=degree,radial_centers_per_subset=centers,
        features=result.shape[1],group_scales=scales,feature_mean=mean.cpu().numpy().tolist(),
        group_masks=masks.tolist(),hierarchy=history,subset_kernel_audits=kernel_audits,
        group_innovation_energy_fraction=fractions,
        block_policy='equal trace for complete polynomial and radial subblocks within each subset',
        full_space_preserved='only lower-order combinations subtracted, then group scaling',
        projection_inputs='W only, all subjects',global_remainder_features_used=False)
