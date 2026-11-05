"""Rotation-plugin discovery, provenance isolation, and subject X/Y alignment."""
import numpy as np
import pytest
from pathlib import Path
from types import SimpleNamespace
from rotations.api import RotationInput, RotationResult, validate_result, skew_matrix
from rotations.engine import discover, BUILTINS, _read_plan, _configs, _digest_json
from rotations.data import aligned_observations, corr


def test_plugin_autodiscovery_without_registry_edits(tmp_path):
    plugin = tmp_path / 'custom_effect.py'
    plugin.write_text('''\nimport numpy as np\nNAME="custom_effect"\nVERSION="1.0"\nDESCRIPTION="custom test"\nDEFAULTS={"alpha":2.0}\ndef objective(data,R,config):\n    return float(config["alpha"]*np.sum(data.rotated(R)**4))\n''',encoding='utf-8')
    plugins, errors = discover([BUILTINS, tmp_path])
    assert not errors
    assert 'custom_effect' in plugins
    assert plugins['custom_effect'].defaults == {'alpha':2.0}
    args=SimpleNamespace(seed=11,starts=None,maxiter=None,config_file=None,set=['custom_effect.alpha=3.5'])
    assert _configs(plugins['custom_effect'],args,{'method':'custom_effect'})['alpha']==3.5
    assert _read_plan(None,['custom_effect'])==[{'method':'custom_effect'}]
    assert _digest_json({'a':1,'b':2})==_digest_json({'b':2,'a':1})


def test_optional_dependency_isolated_from_other_plugins(tmp_path):
    (tmp_path/'missing_pkg.py').write_text('import fictional_uninstalled_module_9283128\nNAME="missing_pkg"\n',encoding='utf-8')
    plugins, failed=discover([BUILTINS,tmp_path])
    assert 'missing_pkg' in failed
    assert 'kurtosis' in plugins and 'xy_effect' in plugins


def test_plugin_never_receives_theta():
    data=RotationInput(np.ones((8,4)),np.zeros((8,3,2)),np.zeros((8,3)),np.ones((8,3),bool))
    assert not hasattr(data,'theta')
    with pytest.raises(ValueError):
        RotationInput(data.q,np.zeros((7,3,2)),data.y,data.mask)
    with pytest.raises(ValueError):
        validate_result(RotationResult(np.zeros((4,4))),4)
    with pytest.raises(ValueError):
        validate_result(RotationResult(np.diag([-1.,1.,1.,1.])),4)


def test_variable_length_subject_alignment():
    data=SimpleNamespace(trial_counts=np.array([2,1]),
                         subject_index=np.array([0,0,1]),
                         x=np.array([[1.,3.],[2.,4.],[5.,6.]]),
                         y=np.array([[1],[0],[1]]))
    x,y,mask=aligned_observations(data)
    assert x.shape==(2,2,2) and y.shape==(2,2) and mask.shape==(2,2)
    assert mask.tolist()==[[True,True],[True,False]]
    assert np.array_equal(x[1,0],[5.,6.]) and y[0,1]==0


def test_multifile_method_package_and_transitive_code_hash(tmp_path):
    from rotations.engine import discover_paths, load_plugin, method_code_sha
    pkg = tmp_path/'compositional_mdl'
    pkg.mkdir()
    (pkg/'helper.py').write_text('''def penalty(u):\n    return float((u*u).mean())\n''', encoding='utf-8')
    (pkg/'__init__.py').write_text('''from .helper import penalty\nNAME="compositional_mdl"\nVERSION="0.1"\nDESCRIPTION="multifile package"\nDEFAULTS={"cost":1.0}\ndef objective(data,R,config):\n    return config["cost"]*penalty(data.rotated(R))\n''',encoding='utf-8')
    path = discover_paths([tmp_path])['compositional_mdl']
    p = load_plugin(path)
    q = np.random.default_rng(10).normal(size=(4, 2))
    data=RotationInput(q,np.zeros((4,1,1)),np.zeros((4,1)),np.ones((4,1),bool))
    assert np.isfinite(p.module.objective(data,np.eye(2),p.defaults))
    old_hash = method_code_sha(path)
    (pkg/'helper.py').write_text('def penalty(u):\n    return float(2*(u*u).mean())\n',encoding='utf-8')
    assert method_code_sha(path) != old_hash


def test_run_persists_matrix_U_and_preserves_theta_boundary(tmp_path):
    from sklearn.decomposition import PCA
    from rotations.engine import _run_one, discover
    rng = np.random.default_rng(17)
    z = rng.normal(size=(26, 4))
    tr = np.arange(15)
    te = np.arange(20,26)
    va = np.arange(15,20)
    pc = PCA(n_components=4).fit(z[tr])
    W = pc.components_ / np.sqrt(np.maximum(pc.explained_variance_,1e-5))[:,None]
    chosen = tr[:9]
    q = (z - pc.mean_) @ W.T
    x = rng.normal(size=(len(chosen), 2, 3))
    y = rng.integers(0,2,size=(len(chosen),2))
    mask = np.ones_like(y,bool)

    class NoTheta:
        @property
        def parameters(self):
            raise RuntimeError('Theta accessed during plugin fit')

    source = {'root':tmp_path,'data':RotationInput(q[chosen], x, y, mask),
              'z':z,'pca':pc,'W':W,'chosen':chosen,'tr':tr,'val':va,'test':te,
              'latent_sha':'source-z','dataset_sha':'source-xy',
              'checkpoint_sha':'checkpoint','dataset':NoTheta()}
    args = SimpleNamespace(model='CR3',condition='beta0_L4', repeat=1, representation='full',
                           seed=31,fit_max_subjects=15,whiten_floor=1e-5,
                           no_evaluate=True,overwrite=False,dry_run=False)
    plugins,_=discover([BUILTINS])
    status,out=_run_one(args,plugins['identity'],plugins['identity'].defaults,source)
    assert status=='completed'
    with np.load(out/'rotation__matrix.npz',allow_pickle=False) as f:
        R,A,b=f['R'],f['A'],f['offset']
        assert np.allclose(R,np.eye(4))
    with np.load(out/'rotation__U.npz',allow_pickle=False) as f:
        U=f['U']
    assert np.allclose(U,z@A.T+b)
    assert np.allclose(U[chosen],source['data'].rotated(R))
    assert _run_one(args,plugins['identity'],plugins['identity'].defaults,source)[0]=='skipped'
    # Integration: theta is accessed only AFTER R and U have been persisted.
    class DatasetWithTheta:
        @property
        def parameters(self):
            assert (out/'rotation__matrix.npz').is_file()
            assert (out/'rotation__U.npz').is_file()
            return z + 0.01 * rng.normal(size=z.shape)
    source['dataset'] = DatasetWithTheta()
    args.no_evaluate = False
    status, evaluated = _run_one(args, plugins['identity'], plugins['identity'].defaults, source)
    assert status == 'completed'
    assert (evaluated/'rotation__test_evaluation.json').is_file()
    assert (evaluated/'rotation__test_corr2.png').is_file()
