"""Smoke and mathematical correctness checks, no hidden generator access."""
import sys, json
from pathlib import Path
import numpy as np
import pytest
from scipy.special import expit

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from basis import Basis
from fit import rank_pairs, estimate, load_data
from scan import givens,complexity,scan
from compare import angle_error


def test_basis_derivatives_for_nonlinear_z():
    rng=np.random.default_rng(11)
    W=rng.normal(size=(350,7))
    b=Basis(3,2).fit(W,3)
    z=W[:8,:3]
    exact=b.jacobian_z(z)
    for i in range(3):
        pp=z.copy();mm=z.copy();pp[:,i]+=1e-5;mm[:,i]-=1e-5
        zp=b._poly((pp-b.zmean)/b.zsd,b.zexp)
        zm=b._poly((mm-b.zmean)/b.zsd,b.zexp)
        assert np.allclose((zp-zm)/(2e-5)*b.zsd[i],exact[:,i,:],atol=3e-5)


def test_link_invariant_training_signs():
    rng=np.random.default_rng(12)
    y=rng.normal(size=250)
    a=expit(y);b=np.logaddexp(0,y)
    i,j,s=rank_pairs(a,1000,np.random.default_rng(1))
    ii,jj,ss=rank_pairs(b,1000,np.random.default_rng(1))
    assert np.array_equal(i,ii) and np.array_equal(j,jj)
    assert np.array_equal(s,ss)


def test_c1_structural_axes_not_rotated():
    rng=np.random.default_rng(15)
    W=rng.normal(size=(500,7))
    basis=Basis(1,2).fit(W,4)
    # C1 true index z1+sqrt(3)*z2*x1+sqrt(3)*z3*x2+sqrt(3)*z4*x3
    z=W[:,:4];x=W[:,4:]
    eta=z[:,0]+np.sqrt(3)*np.sum(z[:,1:]*x,axis=1)
    reg=np.linalg.lstsq(np.c_[np.ones(len(W)),basis.transform(W)],eta,rcond=None)[0]
    B=basis.coefficient_matrix(reg[1:])
    G=np.einsum('nit,tk->nik',basis.jacobian_z(z[:60]),B)
    s0=np.sqrt(np.mean(G**2,axis=0))
    rotated=np.einsum('ij,njk->nik',givens(4,1,2,45.),G)
    s45=np.sqrt(np.mean(rotated**2,axis=0))
    assert complexity(s0)['normalized_l1'] < complexity(s45)['normalized_l1']


def test_cr3_discrete_design_basis_without_perturbing_x(tmp_path):
    rng=np.random.default_rng(8)
    z=np.c_[rng.uniform(.4,1.,400),rng.uniform(.5,3.,400),
            rng.uniform(.2,1.2,400),rng.uniform(.6,3.8,400)]
    bank=np.array([[5.,0.,.1,10.,5.,.5],[100.,0.,.8,20.,5.,.2],
                   [0.,15.,.2,50.,0.,.9],[30.,20.,.4,5.,0.,.5]])
    x=bank[rng.integers(len(bank),size=len(z))]
    W=np.c_[z,x]
    basis=Basis(2,2).fit(W,4)
    assert basis.transform(W).shape[1]>100
    coef=rng.normal(size=basis.transform(W).shape[1])
    out=tmp_path/'fit';out.mkdir()
    np.savez(out/'coefficients.npz',coef_scaled=coef,anchors=z[:70])
    (out/'fit.json').write_text(json.dumps({'source':'wy_only','model':'CR3','label':'prob',
          'method':'rank_svm','basis':basis.serialize()}))
    result=scan(out,tmp_path/'curve',0,2,15)
    assert len(result['records'])==13
    assert all(np.isfinite(r['normalized_l1']) for r in result['records'])
    assert (tmp_path/'curve'/'summary.json').exists()


def test_rank_estimator_c1_works():
    rng=np.random.default_rng(1)
    W=rng.normal(size=(1050,7))
    eta=W[:,0]+np.sqrt(3)*np.sum(W[:,1:4]*W[:,4:],axis=1)
    y=expit(eta)
    b,c,B,details=estimate(W[:650],y[:650],W[650:850],y[650:850],
                           W[850:],y[850:],4,pairs=1200,z_degree=1,x_degree=1)
    assert B.shape==(4,3)
    assert details['test_rank_accuracy']>.90


def test_binary_requires_explicit_surrogate(tmp_path):
    data=tmp_path/'binary';data.mkdir()
    (data/'meta.json').write_text(json.dumps({'model':'CR3','label':'bernoulli','z_dim':4,'x_dim':6}))
    rng=np.random.default_rng(12)
    for split in ('train','val','test'):
        np.savez(data/(split+'.npz'),W=rng.normal(size=(100,10)),y=rng.integers(0,2,100))
    with pytest.raises(ValueError,match='Bernoulli'):
        load_data(data,100,12)


def test_sign_permutation_error():
    assert angle_error(-45.,45.)<1e-8
    assert angle_error(42.,45.)==3.


def test_gate2_cr3_refuses_bad_oracle_projection(tmp_path):
    from gate2 import evaluate
    oracle=tmp_path/'oracle';(oracle/'scan').mkdir(parents=True)
    (oracle/'fit.json').write_text(json.dumps({'source':'oracle','model':'CR3','test_r2':.51}))
    rec=[{'angle_deg':float(a),'normalized_l1':2.+abs(a)/90.,'effective_partner_sum':3.+abs(a)/90.}
         for a in np.arange(-90,91,5)]
    (oracle/'scan'/'rotation_curve.json').write_text(json.dumps({'source':'oracle','records':rec}))
    result=evaluate(oracle,tmp_path/'gate')
    assert all(r['gate2']=='FAIL' for r in result['scores'].values())
    assert (tmp_path/'gate'/'gate2.json').exists()
