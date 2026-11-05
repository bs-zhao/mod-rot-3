"""Method-specific smoke test: identity."""
import numpy as np
from rotations.${name}.api import RotationInput
import rotations.identity as method

def test_identity_smoke():
    rng = np.random.default_rng(12)
    q = rng.normal(size=(16,4))
    x = rng.normal(size=(16,5,3))
    y = rng.integers(0,2,size=(16,5))
    data = RotationInput(q,x,y,np.ones_like(y,dtype=bool))
    result = method.fit(data,{})
    assert np.array_equal(result.R,np.eye(4))
