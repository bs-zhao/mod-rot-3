"""Direct mathematical and end-to-end tests for effect_fingerprint."""
from types import SimpleNamespace
import numpy as np
from sklearn.decomposition import PCA
from .metrics import FingerprintObjective
from .optimizer import as_rotation, optimize
from .surrogate import make_proxy
from .stability import best_match_rows, plane_curves
from . import fit
from . import experiment


def data(n=96,t=20,k=3):
    rng=np.random.default_rng(711)
    q=rng.normal(size=(n,k))
    x=rng.normal(size=(n,t,3))
    eta=.7*q[:,0,None]*x[:,:,0]+.4*q[:,1,None]**2*x[:,:,1]-.4*x[:,:,2]
    y=(rng.uniform(size=(n,t))<1/(1+np.exp(-eta))).astype(int)
    return q,x,y,np.ones_like(y,dtype=bool)


def cfg(**kw):
    import json
    from pathlib import Path
    c=json.loads((Path(__file__).parent/"config.json").read_text())
    c.update(starts=1,maxiter=3,anchor_subjects=40,plane_steps=5,
             plane_pairs=1,make_plots=False,stress_enabled=False)
    c.update(kw)
    return c


class Analytic:
    def __init__(self,kind,q):
        self.kind=kind;self.k=2
        self.fit_ids=np.arange(len(q))
        self.q_train=q
    def gradients(self,q):
        if self.kind=="mixed":
            out=np.zeros((len(q),2,2))
            out[:,0,0]=2*q[:,0]
            out[:,1,1]=2*q[:,1]
            return out
        if self.kind=="linear":
            g=np.zeros((len(q),2,2))
            g[:,0,0]=1;g[:,1,1]=1
            return g
        if self.kind=="radial":
            g=np.zeros((len(q),2,1))
            g[:,0,0]=2*q[:,0]
            g[:,1,0]=2*q[:,1]
            return g
        raise ValueError()


def test_two_distinct_nonlinear_stimulus_modes_identifiable():
    rng=np.random.default_rng(1)
    q=rng.normal(size=(220,2))
    a=Analytic("mixed",q)
    o=FingerprintObjective(a,cfg(rank_metric="entropy",channel_mode="raw"))
    rotation=as_rotation([np.pi/4],2)
    assert o.value(np.eye(2)) < o.value(rotation)-.15


def test_linear_and_radial_can_be_nonidentifiable():
    rng=np.random.default_rng(2)
    q=rng.normal(size=(300,2))
    rot=as_rotation([np.pi/4],2)
    o=FingerprintObjective(Analytic("linear",q),cfg())
    assert abs(o.value(np.eye(2))-o.value(rot))<1e-8
    # A quadratic function of r^2 is rotationally symmetric: fingerprint
    # differences of small finite samples are not evidence of identification.
    p=FingerprintObjective(Analytic("radial",q),cfg())
    assert abs(p.value(np.eye(2))-p.value(rot))<.13


def test_rank_definitions_and_channel_modes_are_finite():
    q,x,y,mask=data()
    proxy=make_proxy(q,x,y,mask,cfg())
    for rank in ("entropy","participation","tail"):
        for mode in ("raw","unitvar","whiten"):
            o=FingerprintObjective(proxy,cfg(rank_metric=rank,channel_mode=mode))
            R=as_rotation(np.array([.3,-.1,.2]),3)
            assert np.isfinite(o.value(R))
            assert np.allclose(R@R.T,np.eye(3),atol=1e-8)
            assert len(o.components(R)["ranks"])==3


def test_fit_saves_valid_rotation():
    q,x,y,mask=data()
    R,info,_=fit(q,x,y,mask,cfg())
    assert R.shape==(3,3)
    assert np.allclose(R@R.T,np.eye(3),atol=1e-7)
    assert info["surrogate_validation"]["fit_subjects"]>0
    assert best_match_rows(R,R)["mean_abs_axis_cos"]>1-1e-8


def test_end_to_end_truth_is_postfit_only(tmp_path):
    q,x,y,mask=data(n=108,k=4)
    tr=np.arange(82);va=np.arange(82,94);te=np.arange(94,108)
    pca=PCA(n_components=4).fit(q[tr])
    W=pca.components_/np.sqrt(np.maximum(pca.explained_variance_,1e-5))[:,None]
    white=(q-pca.mean_)@W.T
    chosen=tr[:72]
    root=tmp_path/"rotations"/"effect_fingerprint"/"synthetic"/"repeat01"
    class AuditTruth:
        @property
        def parameters(self):
            assert len(list(root.glob("run_*/rotation__U.npz")))==1
            path=next(root.glob("run_*"))
            assert (path/"rotation__fit.json").exists()
            return q.copy()
    source=dict(output_root=tmp_path,z=q,chosen=chosen,q_fit=white[chosen],
                x_fit=x[chosen],y_fit=y[chosen],mask_fit=mask[chosen],pca=pca,W=W,
                tr=tr,val=va,te=te,native_k=4,rotation_dim=4,dataset=AuditTruth(),
                dataset_sha="fake",latent_sha="fake-z",checkpoint_sha="fake-ckpt")
    args=SimpleNamespace(model="CR3",condition="synthetic",repeat=1,
                         representation="pca",rotation_dim=4,fit_max_subjects=72,
                         whiten_floor=1e-5,seed=31,no_evaluate=False,
                         dry_run=False,overwrite=False)
    path=experiment.execute(args,cfg(),fit,source)
    from . import postfit
    postfit(path,cfg())
    assert (path/"fingerprint__diagnostics.json").is_file()
    assert (path/"fingerprint__plane_curves.csv").is_file()
    assert (path/"rotation__test_evaluation.json").is_file()
    with np.load(path/"rotation__matrix.npz") as f:
        assert np.allclose(f["R"]@f["R"].T,np.eye(4),atol=1e-7)



def test_stress_retrains_only_fit_subjects():
    from .stability import stress_runs
    q,x,y,mask=data(n=90,k=3)
    config=cfg(stress_subject_counts=[64],stress_seeds=[12],
               stress_rank_metrics=["participation"],stress_channel_modes=["raw"],
               stress_starts=1,stress_maxiter=2)
    proxy=make_proxy(q,x,y,mask,config)
    results=stress_runs(proxy,(q,x,y,mask),np.eye(3),config)
    assert len(results)==1
    record=results[0]
    assert record["subjects"]==64
    assert 0<=record["alignment"]["mean_abs_axis_cos"]<=1
    assert np.allclose(record["R"]@record["R"].T,np.eye(3),atol=1e-7)


def test_cross_beta_comparison_aligns_axes_on_train_only(tmp_path):
    from .compare import compare_pair
    rng=np.random.default_rng(111)
    ua=rng.normal(size=(120,4))
    ub=ua[:,[2,0,3,1]]*np.array([-1,1,-1,1])
    ub+=rng.normal(0,.005,size=ub.shape)
    common=dict(source_sha="same",train=np.arange(80),
                test=np.arange(80,120),matched_mean_r2=.55)
    a=dict(common,U=ua,folder=tmp_path/"a")
    b=dict(common,U=ub,folder=tmp_path/"b")
    result=compare_pair(a,b)
    assert result["train_axis_mean_abs_corr"]>.999
    assert result["heldout_test_axis_mean_signed_corr"]>.999
    b["source_sha"]="other"
    import pytest
    with pytest.raises(ValueError,match="Different S1"):
        compare_pair(a,b)



def test_windows_plain_string_config_override():
    from .experiment import cli
    _,c=cli(["--model","CR3","--condition","synthetic",
             "--set","rank_metric=participation",
             "--set","channel_mode=unitvar"])
    assert c["rank_metric"]=="participation"
    assert c["channel_mode"]=="unitvar"
