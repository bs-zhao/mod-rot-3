from pathlib import Path
import sys, tempfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import build
from models import fit,load_fit
from interactions import detect
from oracle import evaluate
from rotation import givens,evaluate_rotation


def test_end_to_end_toy():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        meta=build('toy_b',str(root/'data'),n_subjects=30,n_trials=12,label='prob')
        assert meta['z_dim']==3
        fit(str(root/'data'),'spline_add',str(root/'model'),max_train=500)
        m,_=load_fit(root/'model')
        with np.load(root/'data'/'test.npz') as test:
            W=test['W'].astype(float)
        pred=m.predict
        estimates,detail=detect(pred,W,3,'toy_b',n_eval=12)
        truth,_=evaluate(root/'data',root/'oracle',scale='prob',n_eval=12)
        assert estimates.shape==truth.shape==(5,5)
        assert np.isfinite(estimates).all()
        R=givens(3,0,1,45)
        rotated,_=evaluate_rotation(pred,W,R,3,'toy_b','fd',12,31)
        assert rotated.shape==truth.shape


def test_oracle_toy_rotational_identity():
    from oracle import logits
    rng=np.random.default_rng(12)
    z=rng.normal(size=(30,3));z[:,2]=rng.uniform(.3,1.8,30)
    x=rng.uniform(.3,1.8,(30,2)); W=np.c_[z,x]
    g=logits('toy_b',W,3)
    u1=(z[:,0]+z[:,1])/np.sqrt(2);u2=(z[:,0]-z[:,1])/np.sqrt(2)
    direct=np.sqrt(2)*u1*x[:,0]**z[:,2]+2*u1*u2*x[:,1]**z[:,2]
    np.testing.assert_allclose(g,direct,rtol=1e-12,atol=1e-12)


def test_cr3_invalid_coordinates():
    from interactions import detect
    rng=np.random.default_rng(9)
    w=rng.normal(size=(20,10))
    m,d=detect(lambda x:x[:,0]*x[:,1]+x[:,4],w,4,'CR3',n_eval=10)
    assert np.isnan(m[4:, :]).all() and np.isnan(m[:, 4:]).all()
    assert len(d['valid_stimulus_z_modulation'])==4
