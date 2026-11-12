import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import norm_lift_profiles as lift


def test_lift_counts_sign_reversals_that_ordinary_matrix_rank_misses():
    direction=np.array([1.,2.,-3.]);positive=np.linspace(.1,2.,60)
    both=positive.copy();both[::2]*=-1
    g=np.stack((positive[:,None]*direction,both[:,None]*direction),axis=1)
    values,error=lift.spectra(torch.tensor(g),np.eye(2))
    assert error<1e-12
    assert values['ordinary'][1,1]<1e-20
    assert values['norm_lift'][0,1]<1e-20
    assert values['norm_lift'][1,1]>.1*values['norm_lift'][1].sum()
    np.testing.assert_allclose(values['ordinary'].sum(1),values['norm_lift'].sum(1),atol=1e-12)


def test_lift_is_invariant_to_output_basis_axis_sign_and_coordinate_registration():
    rng=np.random.default_rng(894);g=rng.normal(size=(80,4,5));axes=np.linalg.qr(rng.normal(size=(4,4)))[0]
    output=np.linalg.qr(rng.normal(size=(5,5)))[0];registration=np.linalg.qr(rng.normal(size=(4,4)))[0]
    first,_=lift.spectra(torch.tensor(g),axes.T)
    changed=np.einsum('nar,ab,rs->nbs',g,registration,output)
    second,_=lift.spectra(torch.tensor(changed),(-registration.T@axes).T)
    for key in first:np.testing.assert_allclose(first[key],second[key],atol=1e-12)
