import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import expit
from core import (gradient_direction_strength,ordinal_contrast_strength,
                  orthogonal_plane,transform,inverse,signed_permutation_angle_error)

def toy(W):
    a,b,c,x1,x2=W.T
    return (a+b)*x1**c+(a*a-b*b)*x2**c

def sample(seed=7,n=600):
    rng=np.random.default_rng(seed)
    z=np.c_[rng.normal(0,.7,(n,2)),rng.uniform(.7,1.3,n)]
    x=rng.uniform(.5,1.5,(n,2))
    return np.c_[z,x]

def test_rotation_inverse():
    W=sample();R=orthogonal_plane(3,0,1,45)
    assert np.allclose(W,inverse(transform(W,R,3,np.zeros(3),np.ones(3)),R,3,np.zeros(3),np.ones(3)))

def test_link_invariant_gradient_direction():
    W=sample(n=200)
    step_x=np.array([.0002,.0002]);step_z=np.array([.004,.004,.004])
    x,_=gradient_direction_strength(toy,W,3,step_x,step_z)
    p,_=gradient_direction_strength(lambda w:expit(toy(w)),W,3,step_x,step_z)
    assert np.allclose(x,p,rtol=.012,atol=.012), (x,p)
    assert np.isfinite(p).all()

def test_link_invariant_ordinal():
    W=sample(n=150)
    z=W[:,:3];xa=W[:,3:];xb=xa[::-1].copy()
    a,_=ordinal_contrast_strength(toy,z,xa,xb,np.full(3,.3))
    b,_=ordinal_contrast_strength(lambda w:expit(toy(w)),z,xa,xb,np.full(3,.3))
    assert np.allclose(a,b)

def test_angle_equivalence():
    assert signed_permutation_angle_error(-45,45)==0
    assert signed_permutation_angle_error(40,-50)==0
