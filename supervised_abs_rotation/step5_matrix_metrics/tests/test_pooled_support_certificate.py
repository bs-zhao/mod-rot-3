import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pooled_support_certificate as certificate


def test_concavity_bound_holds_for_every_tested_rotation_and_is_tight_for_a_common_eigenbasis():
    rng=np.random.default_rng(821);q=np.linalg.qr(rng.normal(size=(4,4)))[0]
    eigen=rng.uniform(.01,1.,size=(3,4))
    h=np.array([q@np.diag(e)@q.T for e in eigen]);h=certificate.normalized_matrices(h)
    bound,_,_=certificate.lower_bound(h)
    np.testing.assert_allclose(certificate.score(h,q),bound,atol=1e-12)
    for _ in range(40):
        trial=np.linalg.qr(rng.normal(size=(4,4)))[0]
        assert certificate.score(h,trial)>=bound-1e-12


def test_fine_grid_records_the_actual_15_degree_minimum_without_a_baseline_veto():
    a=np.deg2rad(15);q=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
    h=certificate.normalized_matrices(np.array([q@np.diag([1.,.001])@q.T,q@np.diag([.002,1.])@q.T]))
    rows=certificate.fine_plane_grid(h,np.eye(2))
    best=min(rows,key=lambda r:r['loss'])
    assert best['angle_deg']==15 and best['loss']<certificate.score(h,np.eye(2))
