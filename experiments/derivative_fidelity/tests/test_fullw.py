import numpy as np
from experiments.derivative_fidelity.metrics_extended import fullw_fields, summarize_fullw

def test_fullw_permutations():
    shape=(2,3)
    fields={
      "U":np.ones((*shape,2)),
      "X":np.ones((*shape,1)),
      "UU":np.ones((*shape,2,2))*2,
      "UX":np.ones((*shape,2,1))*3,
      "XX":np.ones((*shape,1,1))*4,
      "UUU":np.ones((*shape,2,2,2))*5,
      "UUX":np.ones((*shape,2,2,1))*6,
      "UXX":np.ones((*shape,2,1,1))*7,
      "XXX":np.ones((*shape,1,1,1))*8,
    }
    arrays=fullw_fields(fields)
    h=arrays["WW"]
    t=arrays["WWW"]
    assert h.shape==(2,3,3,3)
    assert np.allclose(h[:,:,0,2],3)
    assert np.allclose(h[:,:,2,0],3)
    assert np.allclose(t[:,:,0,1,2],6)
    assert np.allclose(t[:,:,0,2,1],6)
    assert np.allclose(t[:,:,2,0,1],6)
    assert np.allclose(t[:,:,0,2,2],7)
    assert np.allclose(t[:,:,2,0,2],7)
    assert np.allclose(t[:,:,2,2,0],7)
