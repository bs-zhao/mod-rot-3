"""Cheap baseline contract checks; full integration runs use real 5k-subject data."""
from __future__ import annotations
import numpy as np
from modlearn_rot.generators.registry import get_model
from modlearn_rot.nn.np.point.discrete import DiscreteNPConfig, DiscreteNeuralProcess
from modlearn_rot.training.np.point.discrete import DiscreteNPTrainingConfig


def selftest():
    for name, dim in [('C1',4),('C2',4),('C3',6)]:
        model=get_model(name)
        rng=np.random.default_rng(11)
        params=np.asarray(model.sample_parameters(4,rng))
        stimuli=np.asarray(model.sample_inputs(7,rng))
        assert params.shape==(4,dim), (name,params.shape)
        assert stimuli.ndim==2 and stimuli.shape[0]==7
        prob=np.asarray(model.choice_probability(stimuli,params[0]))
        assert prob.shape==(7,) and np.all((prob>0)&(prob<1))
        config=DiscreteNPConfig(x_dim=stimuli.shape[1],latent_dim=dim)
        net=DiscreteNeuralProcess(config)
        assert net.config.latent_dim==dim
        cfg=DiscreteNPTrainingConfig(beta=0,lambda_sparse=0)
        assert cfg.context_min_fraction==.25
        print(name,'unchanged generator / NP architecture PASS')
    print('Baseline model selftest PASS (does not regenerate datasets)')