"""Method-specific smoke test: xy_effect."""
import numpy as np
from rotations.${name}.api import RotationInput
import rotations.xy_effect as method

def test_xy_effect_smoke():
    rng = np.random.default_rng(12)
    q = rng.normal(size=(16,4))
    x = rng.normal(size=(16,5,3))
    y = rng.integers(0,2,size=(16,5))
    data = RotationInput(q,x,y,np.ones_like(y,dtype=bool))
    result = method.fit(data,{**method.DEFAULTS,"starts":1,"maxiter":2})
    assert result.R.shape==(4,4)
    assert np.allclose(result.R@result.R.T,np.eye(4),atol=1e-7)
