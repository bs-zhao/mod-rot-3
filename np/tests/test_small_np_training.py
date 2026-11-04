"""Train one epoch of the exact frozen discrete NP; assert full-trial encoding."""
import numpy as np
import torch
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig
from modlearn_rot.training.np.point.discrete import (
    DiscreteData, DiscreteNPTrainer, DiscreteNPTrainingConfig,
)
from modlearn_rot.training.continuous import split_subject_indices, make_fixed_context_masks


def test_training_and_all_trial_encoding_are_compatible():
    torch.set_num_threads(1)
    n,t,d,k=18,12,3,4
    g=torch.Generator().manual_seed(19)
    x=torch.randn((n,t,d),generator=g)
    choice=torch.randint(0,2,(n,t),generator=g)
    data=DiscreteData(x=x,choice=choice,trial_mask=torch.ones((n,t),dtype=torch.bool))
    cfg=DiscreteNPTrainingConfig(beta=0.0,lambda_sparse=0.0,
                                 batch_size=6,max_epochs=1,patience=1,
                                 context_min_fraction=.25,context_max_fraction=.8,
                                 seed=19)
    splits=split_subject_indices(n,cfg.train_fraction,cfg.validation_fraction,31)
    masks=make_fixed_context_masks(data,splits,.25,.8,31)
    result=DiscreteNPTrainer(DiscreteNPConfig(x_dim=d,latent_dim=k),cfg,'cpu').fit(
        data,subject_splits=splits,fixed_context_masks=masks)
    assert np.isfinite(result['test_metrics']['reconstruction'])
    model=result['model'].eval()
    mean=result['preprocessing']['encoder_mean'].flatten()
    std=result['preprocessing']['encoder_std'].flatten()
    obs=(torch.cat((x,choice.float().unsqueeze(-1)),dim=-1)-mean)/std
    with torch.no_grad():
        z,lv=model.encode_context(obs[...,:d],obs[...,d:],data.trial_mask)
    assert z.shape==(n,k) and lv.shape==(n,k)
    assert torch.isfinite(z).all() and torch.isfinite(lv).all()