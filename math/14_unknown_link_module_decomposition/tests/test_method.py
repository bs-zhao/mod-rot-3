import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from basis import Dictionary, givens, complexity
from link import hinge_basis, learn_monotone_transform, apply
from scan import get_targets, REV
from gate import angle_distance


def test_rotation_so_d():
    for theta in range(-90,91,5):
        R=givens(4,0,2,theta)
        assert np.allclose(R.T @ R,np.eye(4),atol=1e-12)
        assert abs(np.linalg.det(R)-1)<1e-12
    assert np.allclose(givens(3,0,1,90)@givens(3,0,1,90).T,np.eye(3))


def test_givens_changes_both_sides():
    R=givens(3,0,1,45)
    assert np.isclose(R[0,1],2**-.5)
    assert np.isclose(R[1,0],-2**-.5)


def test_c1_true_axis_sparse():
    rng=np.random.default_rng(42);z=rng.normal(size=(120,4));x=rng.normal(size=(120,3))
    dic=Dictionary.fit(np.column_stack([z,x]),4,1,1)
    # C1 coefficient matrix for [b1,b2,b3,b4] × [x1,x2,x3].
    coef=np.zeros((4,3));coef[1,0]=coef[2,1]=coef[3,2]=1
    s0=dic.groups_from_interactions(coef.ravel())
    R=givens(4,1,2,45)
    s45=dic.groups_from_interactions((R@coef).ravel())
    assert complexity(s0)['module_l21'] < complexity(s45)['module_l21']


def test_cr3_discrete_X_no_perturb():
    rng=np.random.default_rng(9)
    X=rng.choice([0,10,40,100],(80,6)).astype(float)
    W=np.column_stack([rng.normal(size=(80,4)),X])
    dic=Dictionary.fit(W,4,zdegree=2,xdegree=2)
    F,sh=dic.features(W,givens(4,0,2,25))
    assert sh==(14,27)
    assert F.shape==(80,14+27+14*27)
    assert np.array_equal(W[:,4:],X)
    assert len(dic.xgroup)==27


def test_link_piecewise_monotonicity_and_normalization():
    rng=np.random.default_rng(1)
    x=rng.normal(size=600);y=1/(1+np.exp(-x))
    F=np.column_stack([x,x*x]);link=learn_monotone_transform(y,F,n_segments=8)
    g=np.linspace(min(y),max(y),200)
    assert np.all(np.diff(apply(g,link))>0)
    assert np.isclose(apply(np.array([min(y),max(y)]),link)[-1]-apply(np.array([min(y),max(y)]),link)[0],1)


def test_link_constant_collapse_blocked():
    rng=np.random.default_rng(42);y=np.sort(rng.uniform(.02,.98,300))
    L=learn_monotone_transform(y,rng.normal(size=(300,3)),n_segments=8)
    assert L['min_relative_slope'] >= .05-1e-7


def test_binary_no_oracle_prob_leak():
    rng=np.random.default_rng(4)
    splits={'train':(rng.normal(size=(150,10)),rng.choice([0.,1.],150)),
            'val':(rng.normal(size=(80,10)),rng.choice([0.,1.],80)),
            'test':(rng.normal(size=(80,10)),rng.choice([0.,1.],80))}
    with pytest.raises(ValueError,match='require --source-fit'):
        get_targets(Path('.'),{'model':'CR3','label':'bernoulli'},splits,'wy')
    with pytest.raises(ValueError,match='requires true parameter'):
        get_targets(Path('.'),{'latent_source':'leaky'},splits,'oracle')


def test_signed_permutation_angle():
    assert angle_distance(-90,0) < 1e-9
    assert angle_distance(45,-45) < 1e-9
    assert angle_distance(-40,45)==5


def test_version_tag_and_group_complexity():
    assert REV.startswith('orthogonal_givens_v3')
    assert complexity(np.zeros((4,5)))['module_l21'] is None


def test_profile_link_invariant_to_rotation_of_full_sieve():
    rng=np.random.default_rng(11)
    W=np.column_stack([rng.normal(size=(320,4)),rng.uniform(-1,1,size=(320,3))])
    y=1/(1+np.exp(-(W[:,0]+.7*W[:,1]*W[:,4]+.2*W[:,2]*W[:,5]**2)))
    dic=Dictionary.fit(W,4,1,2)
    f0,_=dic.features(W)
    fr,_=dic.features(W,givens(4,0,1,37))
    link0=learn_monotone_transform(y,f0,n_segments=8)
    linkr=learn_monotone_transform(y,fr,n_segments=8)
    assert np.allclose(link0['slopes'],linkr['slopes'],atol=1e-5)
