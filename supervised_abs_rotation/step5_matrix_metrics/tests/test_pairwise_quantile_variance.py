import sys
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairwise_quantile_variance import PairwiseQuantileVariance
from run_quantile_conditional_variance import parse_arguments


def test_pairwise_covariance_matches_explicit_cross_pairs_and_keeps_signed_conservation(tmp_path):
    rng=np.random.default_rng(952);z=rng.normal(size=(50,3));y=rng.normal(size=(50,5));y-=y.mean(0)
    labels=np.array([0]+[1]*4+[2]*10+[3]*35)
    with threadpool_limits(2):
        torch.set_num_threads(2)
        model=PairwiseQuantileVariance(z,torch.tensor(y),tmp_path,'test',device='cpu')
        actual,audit=model.proper_partition_covariance(labels)
        expected=np.zeros((5,5))
        for group in np.unique(labels):
            ids=np.flatnonzero(labels==group)
            if len(ids)<2:continue
            for i in ids:
                for j in ids:
                    if i!=j:expected+=np.outer(y[i],y[j])/(len(y)*(len(ids)-1))
        np.testing.assert_allclose(actual,expected,atol=1e-14)
        assert audit['subject_fraction_with_available_pairs']==49/50
        effects,numerical=model.frame_covariances(np.eye(3))
        np.testing.assert_allclose(effects.sum(0),y.T@y/len(y),atol=1e-12)
        assert not model.PSD_increments_expected
        perm=[2,0,1];other,_=model.frame_covariances(np.eye(3)[:,perm]*np.array([-1,1,-1]))
        np.testing.assert_allclose(other,np.concatenate((effects[perm],effects[-1:])),atol=1e-12)


def test_cli_accepts_pair_correction_and_latest_frame_flags_together():
    args=parse_arguments(['--model','CR3','--run-tag','CI','--audit-latest','--pairwise-correction'])
    assert args.pairwise_correction and args.audit_latest and args.run_tag=='CI'
