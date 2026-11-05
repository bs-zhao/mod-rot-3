"""Method-specific smoke test: quartic."""
import numpy as np
from rotations.quartic.api import RotationInput
import rotations.quartic as method

def test_quartic_smoke():
    rng = np.random.default_rng(12)
    q = rng.normal(size=(16,4))
    x = rng.normal(size=(16,5,3))
    y = rng.integers(0,2,size=(16,5))
    data = RotationInput(q,x,y,np.ones_like(y,dtype=bool))
    loss = method.objective(data,np.eye(4),method.DEFAULTS)
    assert np.isfinite(loss)
