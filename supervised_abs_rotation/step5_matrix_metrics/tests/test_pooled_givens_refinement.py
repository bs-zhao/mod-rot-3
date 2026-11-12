import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pooled_givens_refinement as givens
import pooled_support_certificate as certificate


def test_fast_line_scores_match_actual_rotated_matrix_scores():
    rng=np.random.default_rng(831);a=rng.normal(size=(4,5,5));h=certificate.normalized_matrices(a@a.transpose(0,2,1))
    q=np.linalg.qr(rng.normal(size=(5,5)))[0];angles=np.linspace(-np.pi/4,np.pi/4,17)
    values=givens.line_scores(h,q,1,3,angles)
    for theta,value in zip(angles,values):
        change=np.eye(5);c,s=np.cos(theta),np.sin(theta)
        change[1,1]=change[3,3]=c;change[1,3]=-s;change[3,1]=s
        np.testing.assert_allclose(value,certificate.score(h,q@change),atol=1e-12)


def test_exact_line_refinement_reaches_common_eigenbasis_and_never_increases_loss():
    a=np.deg2rad(13.7);q=np.eye(3);q[:2,:2]=[[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]
    h=certificate.normalized_matrices(np.array([q@np.diag([1.,.001,.01])@q.T,q@np.diag([.002,1.,.1])@q.T]))
    frame,value,history=givens.refine(h,np.eye(3),passes=4)
    bound,_,_=certificate.lower_bound(h)
    assert value<=certificate.score(h,np.eye(3)) and value-bound<1e-8
    np.testing.assert_allclose(frame.T@frame,np.eye(3),atol=1e-12)
    assert np.all(np.diff([r['loss'] for r in history])<=1e-12)


def test_maximum_decrease_first_step_is_at_least_as_good_as_any_initial_plane_grid():
    rng=np.random.default_rng(832);a=rng.normal(size=(3,4,4));h=certificate.normalized_matrices(a@a.transpose(0,2,1))
    q=np.linalg.qr(rng.normal(size=(4,4)))[0]
    grid=np.linspace(-np.pi/4,np.pi/4,361)
    best=min(float(givens.line_scores(h,q,i,j,grid).min()) for i in range(4) for j in range(i+1,4))
    frame,value,_=givens.refine_best_decrease(h,q,steps=1)
    assert value<=best+1e-10
    np.testing.assert_allclose(frame.T@frame,np.eye(4),atol=1e-12)
