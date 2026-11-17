"""End-to-end small synthetic execution without user checkpoint files."""
import sys
import numpy as np
import torch
from types import SimpleNamespace

from experiments.derivative_fidelity import run
from experiments.derivative_fidelity.data import Case


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder=torch.nn.Sequential(
            torch.nn.Linear(7,10),torch.nn.Tanh(),torch.nn.Linear(10,2))

class Truth:
    mu=np.zeros(4)
    sd=np.ones(4)
    def __init__(self,T):
        self.T=np.asarray(T)
    def std(self,ids=None):
        return self.T if ids is None else self.T[ids]


def fake_case():
    rng=np.random.default_rng(12)
    all_v=rng.normal(size=(6,4))
    all_t=all_v.copy()
    X=rng.uniform(-1,1,size=(4,3))
    net=TinyNet().double()
    data={"V":all_v,"train_ids":np.array([0,1,2,3]),
          "mu_z":np.zeros(4),"Minv":np.eye(4),
          "paths":{"checkpoint":"mock.ckpt"}}
    return Case("C1",1,data,Truth(all_t),net,np.eye(4),np.zeros(4),
                np.zeros(4),np.ones(4),np.zeros(3),np.ones(3),
                np.array([4,5]),all_v[[4,5]],all_t[[4,5]],X,"mockhash","shared")


def test_mock_pipeline(tmp_path,monkeypatch):
    monkeypatch.setattr(run,"BASE",tmp_path)
    monkeypatch.setattr(run,"load_case",lambda *a,**k:fake_case())
    args=SimpleNamespace(
        models=["C1"],repeats=[1],channels=["logit"],stage="run",
        tag="mock",subjects=2,stimuli=4,seed=12,grid="shared",
        domain="all",standardized=False,precision="float64",
        device="cpu",beta=None,zdim=None,pca=0,
        blocks="U,UX",methods="ad_mixed",step=.03,sigma=.05,
        mc=8,stein_samples=32,surrogate_train=100,centers=20,epochs=3,
        chunk=4,zero_tol=1e-8,directional_points=4,
        storage="metrics",taylor=False,taylor_points=2,
        force=False,fail_fast=True)
    folder=run.run_case("C1",1,"logit",args)
    assert (folder/"analytic_truth"/"U.npy").exists()
    assert (folder/"analytic_mapped"/"UX.npy").exists()
    assert (folder/"nn_ad64"/"UX.npy").exists()
    assert (folder/"ad_mixed"/"field_comparisons.json").exists()
    assert (folder/"ad_mixed"/"metric_comparisons.jsonl").exists()
    assert (folder/"ad_mixed"/"metric_comparisons_nn_ad.jsonl").exists()
    assert not (folder/"ad_mixed"/"U.npy").exists()  # storage=metrics
    n,m=run.make_report(tmp_path/"mock")
    assert n>0 and m>0
