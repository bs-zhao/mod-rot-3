import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import subset_kernel_group_gp as subset


def test_every_U_subset_has_radial_features_with_W_only_hierarchy_and_unit_trace():
    rng=np.random.default_rng(801);z=rng.normal(size=(200,4))
    with threadpool_limits(2):phi,groups,masks,audit=subset.dictionary(z,3,np.empty((200,0)),device='cpu')
    np.testing.assert_array_equal(masks,np.arange(1,16))
    assert len(audit['subset_kernel_audits'])==15 and audit['global_remainder_features_used'] is False
    for group in range(15):
        np.testing.assert_allclose(np.mean(np.sum(phi[:,groups==group]**2,axis=1)),1.,atol=1e-10)
    assert audit['projection_inputs']=='W only, all subjects'


def test_subset_group_kernels_respect_axis_signs_and_permutations_in_their_Gram_geometry():
    rng=np.random.default_rng(802);z=rng.normal(size=(180,3));permutation=np.array([2,0,1]);signs=np.array([-1.,1.,-1.])
    with threadpool_limits(2):
        a,ag,am,_=subset.dictionary(z,2,np.empty((180,0)),device='cpu')
        b,bg,bm,_=subset.dictionary(z[:,permutation]*signs,2,np.empty((180,0)),device='cpu')
    for i,mask in enumerate(bm):
        original=sum(1<<int(permutation[j]) for j in range(3) if int(mask)&(1<<j))
        j=int(np.flatnonzero(am==original)[0]);x=a[:,ag==j];y=b[:,bg==i]
        np.testing.assert_allclose(x@x.T,y@y.T,rtol=1e-6,atol=1e-8)
