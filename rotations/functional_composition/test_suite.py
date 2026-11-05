"""Independent scientific and integration tests for this single rotation suite."""
from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np
from sklearn.decomposition import PCA
from rotations.functional_composition import fit
from rotations.functional_composition import experiment, metrics
from rotations.functional_composition.optimizer import as_rotation
from rotations.functional_composition.surrogate import make_proxy, rotate_coefficients
from rotations.functional_composition.mdl import prequential_mdl

HERE=Path(__file__).resolve().parent


def cfg(**updates):
    config=json.loads((HERE/'config.json').read_text())
    config.update(dict(anchor_subjects=32,starts=1,maxiter=3,sweep_steps=5,
                       sweep_pairs=1,max_mdl_subjects=100,make_plots=False,
                       ablation_enabled=False,mdl_compare_candidates=False))
    config.update(updates)
    return config


def binary_fake(n=108,t=25,k=3,d=3,seed=17):
    rng=np.random.default_rng(seed)
    q=rng.normal(size=(n,k))
    x=rng.normal(size=(n,t,d))
    eta=(.8*q[:,0,None]*x[:,:,0]+.5*q[:,1,None]*q[:,2,None]*x[:,:,1]
         -0.7*x[:,:,2])
    p=1/(1+np.exp(-eta))
    y=(rng.uniform(size=(n,t))<p).astype(int)
    return q,x,y,np.ones_like(y,dtype=bool)


def test_product_cheaper_than_45_degree_squared_difference():
    # f(a,b)=ab, versus f(u,v)=(u²-v²)/2: one product vs two squares.
    c={0:np.zeros(1),1:np.zeros((2,1)),2:np.zeros((2,2,1))}
    c[2][0,1,0]=c[2][1,0,0]=.5
    R=1/np.sqrt(2)*np.array([[1,1],[-1,1]])
    before=metrics.support_complexity(c,.08)['operator']
    after=metrics.support_complexity(rotate_coefficients(c,R),.08)['operator']
    assert after>before+.5,(before,after)
    assert np.isclose(metrics._tt_rank_control(c),metrics._tt_rank_control(rotate_coefficients(c,R)))


def test_symmetric_tensor_rotation_preserves_predictions():
    q,x,y,mask=binary_fake(n=100)
    proxy=make_proxy(q,x,y,mask,cfg())
    R=as_rotation(np.array([.3,-.4,.2]),3)
    coef=rotate_coefficients(proxy.coef,R)
    u=q@R.T
    predicted=metrics._predict_profiles_from_rotated(coef,u)
    assert np.allclose(predicted,proxy.predict_profiles(q),atol=2e-9)
    assert len(proxy.stimulus.features)+1<=18
    assert np.isfinite(proxy.fidelity['heldout_profile_r2'])


def test_all_twenty_finite_and_orthogonal_fit():
    q,x,y,mask=binary_fake(n=110)
    config=cfg()
    from rotations.functional_composition.surrogate import make_proxy
    from rotations.functional_composition.optimizer import RotationObjective,optimize
    proxy=make_proxy(q,x,y,mask,config)
    R,info,obj=optimize(proxy,config)
    assert np.allclose(R@R.T,np.eye(3),atol=1e-7)
    measured,detail=metrics.diagnostic_metrics(proxy,R,config)
    assert len(measured)==20
    assert set(measured)==set(metrics.NAMES)
    assert all(v is not None and np.isfinite(v) for v in measured.values())
    assert np.isfinite(obj.value(R))
    assert len(detail['axis_fingerprints'])==3


def test_only_own_package_end_to_end_and_truth_only_after_U(tmp_path):
    q,x,y,mask=binary_fake(n=110,k=4)
    config=cfg()
    tr=np.arange(85)
    va=np.arange(85,96)
    te=np.arange(96,110)
    pca=PCA(n_components=4).fit(q[tr])
    W=pca.components_/np.sqrt(np.maximum(pca.explained_variance_,1e-5))[:,None]
    whiten=(q-pca.mean_)@W.T
    chosen=tr[:75]
    base=tmp_path/'rotations'/'functional_composition'/'synthetic'/'repeat01'

    class TruthAfterPersist:
        @property
        def parameters(self):
            assert len(list(base.glob('run_*/rotation__U.npz')))==1
            folder=next(base.glob('run_*'))
            assert (folder/'rotation__matrix.npz').is_file()
            assert (folder/'rotation__fit.json').is_file()
            return q.copy()

    source=dict(root=tmp_path,output_root=tmp_path,z=q,chosen=chosen,
                q_fit=whiten[chosen],x_fit=x[chosen],y_fit=y[chosen],
                mask_fit=mask[chosen],pca=pca,W=W,tr=tr,val=va,te=te,
                native_k=4,rotation_dim=4,dataset=TruthAfterPersist(),
                dataset_sha='d',latent_sha='z',checkpoint_sha='c')
    args=SimpleNamespace(model='CR3',condition='synthetic',repeat=1,
                         representation='pca',rotation_dim=4,fit_max_subjects=75,
                         whiten_floor=1e-5,seed=31,no_evaluate=False,
                         dry_run=False,overwrite=False)
    folder=experiment.execute(args,config,fit,source)
    assert (folder/'rotation__test_evaluation.json').is_file()
    assert (folder/'rotation__test_corr2.png').is_file()
    assert (folder/'rotation__U.npz').is_file()
    with np.load(folder/'rotation__matrix.npz',allow_pickle=False) as file:
        R,A,offset=file['R'],file['A'],file['offset']
    with np.load(folder/'rotation__U.npz',allow_pickle=False) as file:
        U=file['U']
    assert np.allclose(U,q@A.T+offset)
    assert np.allclose(U[chosen],whiten[chosen]@R.T)
    # Diagnostics are separate and do not affect the persisted R.
    from rotations.functional_composition import postfit
    postfit(folder,config)
    assert (folder/'functional__20_metrics.json').is_file()
    assert (folder/'functional__plane_curves.csv').is_file()
    with open(folder/'functional__20_metrics.json') as f:
        result=json.load(f)
    assert len(result['best']['metrics'])==20


def test_noise_calibrated_operator_uses_disjoint_fit_halves():
    from rotations.functional_composition.metrics import calibrated_operator_complexity
    q,x,y,mask=binary_fake(n=112,k=3)
    config=cfg()
    proxy=make_proxy(q,x,y,mask,config)
    assert proxy.split_coefficients is not None
    assert sum(proxy.fidelity['split_half_fit_subjects'])==proxy.fidelity['fit_subjects']
    identity,detail=calibrated_operator_complexity(proxy,np.eye(3),config,details=True)
    tilted=calibrated_operator_complexity(proxy,as_rotation(np.array([.2,.3,-.15]),3),config)
    assert np.isfinite(identity) and np.isfinite(tilted)
    assert 0<=detail['hard_modules']<=detail['total_modules']
    assert detail['calibration'].startswith('FIT')
    assert all(v['uncertainty']>0 for v in detail['modules'])


def test_mdl20_is_nested_and_can_select_nonlinear_modules():
    q,x,y,mask=binary_fake(n=110,k=3)
    config=cfg(mdl_candidate_pool=8,mdl_max_extra_modules=2,
               mdl_selection_trials=12,mdl_prefix_folds=2,
               mdl_min_delta_nats=0.,max_mdl_subjects=85)
    proxy=make_proxy(q,x,y,mask,config)
    mdl=prequential_mdl(proxy,np.eye(3),config)
    assert np.isfinite(mdl['objective'])
    assert np.isfinite(mdl['nonlinear_prediction_gain_nats_per_trial'])
    assert len(mdl['folds'])==2
    assert mdl['selected_module_count']>=3
    assert all(f['inner_fit_subjects']+f['inner_validation_subjects']==f['prefix_subjects']
               for f in mdl['folds'])
    assert all(f['future_trials']>0 and f['selected_nonlinear_modules']<=2
               for f in mdl['folds'])


def test_all_ablation_directions_and_mdl_candidate_selector():
    from rotations.functional_composition.ablation import fit_ablations
    q,x,y,mask=binary_fake(n=112,k=3)
    config=cfg(ablation_enabled=True,ablation_starts=1,ablation_maxiter=2,
               mdl_compare_candidates=True,max_mdl_subjects=80,
               mdl_candidate_pool=5,mdl_max_extra_modules=1,
               mdl_selection_trials=12,mdl_prefix_folds=1)
    proxy=make_proxy(q,x,y,mask,config)
    ab=fit_ablations(proxy,config,np.eye(3))
    assert set(('identity','joint','operator08','fingerprint11','jointdiag13','mdl20_selected'))==set(ab['rotations'])
    assert ab['mdl_winner'] in ('identity','joint','operator08','fingerprint11','jointdiag13')
    assert len(ab['mdl'])==5
    assert all(np.allclose(r@r.T,np.eye(3),atol=1e-7)
               for r in ab['rotations'].values())
