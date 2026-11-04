import numpy as np
from modrot3.analysis import ridge_probe, corr
from modrot3.rotation_api import RotationInput, skew_matrix, validate_result, RotationResult
from modrot3.rotation_engine import discover, BUILTINS, _invoke, _configs, _digest_json
from types import SimpleNamespace


def test_ridge_recovery_heldout():
    rng = np.random.default_rng(10)
    z = rng.normal(size=(150, 4))
    theta = z @ rng.normal(size=(4, 4))
    r = ridge_probe(z, theta, np.arange(100), np.arange(100, 125), np.arange(125, 150))
    assert min(r['test_r2']) > .9999
    assert corr(theta, z).shape == (4, 4)


def test_plugins_and_so4_optimizer():
    plugins, broken = discover([BUILTINS])
    assert not broken
    assert {'identity', 'kurtosis', 'quartic', 'xy_effect'} <= set(plugins)
    rng = np.random.default_rng(13)
    q = rng.laplace(size=(80, 4))
    x = rng.normal(size=(80, 15, 3))
    y = rng.binomial(1, .5, size=(80, 15))
    mask = np.ones((80, 15), bool)
    data = RotationInput(q, x, y, mask)
    for name in ('identity', 'kurtosis', 'quartic', 'xy_effect'):
        cfg = {**plugins[name].defaults}
        if 'starts' in cfg:cfg.update(starts=1, maxiter=3)
        result = _invoke(plugins[name], data, cfg)
        validate_result(result, 4)
        assert np.all(np.isfinite(result.R))
        assert np.isclose(np.linalg.det(result.R), 1.0, atol=1e-6)
    assert np.allclose(skew_matrix(np.zeros(6), 4), np.zeros((4, 4)))


def test_distinct_fingerprints_and_config_override(tmp_path):
    plugins, _ = discover([BUILTINS])
    args = SimpleNamespace(seed=31, starts=4, maxiter=10, config_file=None,
                           set=['kurtosis.maxiter=6'])
    c1 = _configs(plugins['kurtosis'], args, {'method': 'kurtosis'})
    c2 = _configs(plugins['kurtosis'], args, {'method': 'kurtosis', 'config':{'starts':8}})
    assert c1['maxiter'] == 6 and c2['starts'] == 8
    assert _digest_json(c1) != _digest_json(c2)