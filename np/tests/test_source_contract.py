from pathlib import Path
import numpy as np
from modlearn_rot.generators.registry import get_model
from modrot3.sanity import selftest
from modrot3.pipeline import condition
from types import SimpleNamespace


def test_original_binary_generators_and_np():
    selftest()


def test_parameter_priors_and_input_dimensions():
    for model_id,k,d in [('C1',4,3),('C2',4,4),('C3',6,4)]:
        model=get_model(model_id)
        r1=np.random.default_rng(31);r2=np.random.default_rng(31)
        assert np.array_equal(model.sample_parameters(6,r1),model.sample_parameters(6,r2))
        assert model.sample_inputs(8,np.random.default_rng(9)).shape==(8,d)
        assert model.n_factors==k
        assert model.response_family=='discrete_choice'


def test_standard_condition_tag():
    a=SimpleNamespace(beta=0,lambda_sparse=0,context_min=.25,context_max=.8,latent_dim=8)
    assert condition(a)=='beta0_lambda0_ctx0p25to0p8_L8'


def test_original_cr3_reference_preserved():
    import sys
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'reference'/'cr3'))
    from s0_s1_cr3 import COUNTS, dominance_audit
    assert COUNTS=={'gain_only':60,'gamma_neutral':60,'loss_only':40,'general_CR2':340}
    assert dominance_audit(np.array([[10.,0.,.5,10.,0.,.5]]))['total']==0