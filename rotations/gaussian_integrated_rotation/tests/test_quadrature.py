"""Gaussian integration is a distinct mathematical measure, not real Z nodes."""
from types import SimpleNamespace
import json
import numpy as np
import pytest
from scipy.special import ndtr

from rotations.gaussian_integrated_rotation.data import fit_gaussian
from rotations.gaussian_integrated_rotation.quadrature import gaussian_nodes
from rotations.gaussian_integrated_rotation.x_integration import x_nodes


@pytest.mark.parametrize("dim",[3,4])
def test_full_covariance_gaussian_whitens_all_subjects_not_just_marginal_sd(dim):
    rng=np.random.default_rng(100+dim)
    A=rng.normal(size=(dim,dim))
    native=rng.normal(size=(1000,dim))@A.T+3
    fitted=fit_gaussian(native,dim)
    assert fitted["all_subject_count"]==1000
    np.testing.assert_allclose(fitted["mu"],native.mean(0),atol=1e-11)
    np.testing.assert_allclose(fitted["covariance"],np.cov(native,rowvar=False),atol=1e-11)
    np.testing.assert_allclose(np.cov(fitted["q"],rowvar=False),np.eye(dim),atol=1e-10)
    np.testing.assert_allclose(native,fitted["mu"]+fitted["q"]@fitted["B"].T,atol=1e-10)


@pytest.mark.parametrize("dim",[3,4])
def test_sobol_truncated_nodes_have_gaussian_not_uniform_quadrature(dim):
    q,w,meta=gaussian_nodes(dim,"sobol",n=1024,cutoff=2.,seed=20)
    assert q.shape==(1024,dim)
    assert np.all(np.abs(q)<2)
    assert np.isclose(w.sum(),1)
    # E[z^2 ||z|<=a] = 1 - 2a phi(a)/(2Phi(a)-1)
    a=2.
    phi=np.exp(-a*a/2)/np.sqrt(2*np.pi)
    conditional_second=1-(2*a*phi)/(2*ndtr(a)-1)
    assert np.max(abs((q*q).T@w-conditional_second))<.02
    assert meta["distribution"]=="coordinatewise_truncated_N0I"


def test_full_normal_hermite_and_trapezoidal_grid():
    q,w,m=gaussian_nodes(3,"hermite",order=5,cutoff=None)
    assert q.shape==(125,3) and np.isclose(w.sum(),1)
    np.testing.assert_allclose(q.T@w,np.zeros(3),atol=1e-14)
    np.testing.assert_allclose((q*q).T@w,np.ones(3),atol=1e-14)
    q2,w2,_=gaussian_nodes(3,"grid",order=9,cutoff=2)
    assert q2.shape==(9**3,3) and np.isclose(w2.sum(),1)
    assert np.all(((q2*q2).T@w2)>0)
    assert np.all(((q2*q2).T@w2)<1)


def test_no_grid_uniform_unweighted_and_incompatible_hermite_cutoff():
    with pytest.raises(ValueError,match="Gauss-Hermite"):
        gaussian_nodes(4,"hermite",cutoff=2)
    with pytest.raises(ValueError,match="power of two"):
        gaussian_nodes(4,"sobol",n=500)
    with pytest.raises(ValueError,match="Finite grid"):
        gaussian_nodes(3,"grid",cutoff=None)


def _fake_source(tmp_path,model="CR5"):
    root=tmp_path/model
    bank=root/"s0_design"/"fixed500.npz"
    bank.parent.mkdir(parents=True)
    bank.write_bytes(b"bank bytes for hash only")
    x=np.zeros((500,6))
    x[:,[0,1,3,4]]=100
    x[:,[2,5]]=.5
    return dict(model=model,root=root,bank=bank,bank_x=x,
                group_codes=np.repeat(np.arange(4),[60,60,40,340]),
                group_names=["gain_only","gamma_neutral","loss_only","general"],
                hashes=dict(bank="known_bank_sha"))


def test_fixed_bank_is_exact_and_preserves_native_categories(tmp_path):
    source=_fake_source(tmp_path)
    x,w,c,meta=x_nodes(source,"bank",seed=21)
    np.testing.assert_array_equal(x,source["bank_x"])
    assert len(x)==500 and np.all(c==source["group_codes"])
    assert meta["mode"]=="exact_discrete_bank_average"
    assert np.allclose(w,np.ones(500)/500)
    with pytest.raises(ValueError,match="special FIXED X bank"):
        x_nodes(source,"generative",n=16)


def test_cr4_generation_uses_actual_s0_uniform_law_not_bank_distribution(tmp_path):
    source=_fake_source(tmp_path,"CR4")
    source["group_codes"]=np.zeros(500,dtype=int)
    source["hashes"]["bank"]="a_sha"
    meta=dict(independent_X=True,bank_sha256="a_sha",
              amount_range=[5,200],probability_range=[.05,.95])
    (source["bank"].parent/"CR4_3P_design.json").write_text(json.dumps(meta))
    x,w,c,desc=x_nodes(source,"generative",n=256,seed=33)
    assert x.shape==(256,6)
    assert np.all((x[:,[0,1,3,4]]>=5)&(x[:,[0,1,3,4]]<200))
    assert np.all((x[:,[2,5]]>=.05)&(x[:,[2,5]]<.95))
    assert desc["distribution"]=="CR4_actual_S0_six_independent_uniform_generator"
    assert np.isclose(w.sum(),1) and not np.any(c)


def test_explicit_independent_uniform_spec_rejects_invalid_lottery(tmp_path):
    source=_fake_source(tmp_path)
    p=tmp_path/"x_spec.json"
    p.write_text(json.dumps(dict(family="independent_uniform",
                    low=[5,5,-.5,5,5,.1],high=[200,200,.95,200,200,.95])))
    with pytest.raises(ValueError,match="probabilities"):
        x_nodes(source,"generative",n=8,x_spec_file=p)
