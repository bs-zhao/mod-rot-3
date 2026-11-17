"""End-to-end small synthetic execution without user checkpoint files."""
import sys
from pathlib import Path
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

def test_windows_style_locked_mmap_cleanup(tmp_path,monkeypatch):
    """Emulate Windows refusal to unlink any still-memory-mapped .npy file."""
    original_unlink = Path.unlink
    original_load = np.load
    refs = []
    import weakref

    def tracked_load(path,*args,**kwargs):
        result = original_load(path,*args,**kwargs)
        if kwargs.get("mmap_mode") == "r" and isinstance(result,np.memmap):
            refs.append((Path(path),weakref.ref(result)))
        return result

    def windows_unlink(path,*args,**kwargs):
        for target,ref in refs:
            arr=ref()
            if arr is not None and target == Path(path):
                mm=getattr(arr,"_mmap",None)
                if mm is not None and not mm.closed:
                    raise PermissionError("Simulated WinError 32: file is mapped")
        return original_unlink(path,*args,**kwargs)

    monkeypatch.setattr(np,"load",tracked_load)
    monkeypatch.setattr(Path,"unlink",windows_unlink)
    monkeypatch.setattr(run,"BASE",tmp_path)
    monkeypatch.setattr(run,"load_case",lambda *a,**k:fake_case())
    from types import SimpleNamespace
    args=SimpleNamespace(
        models=["C1"],repeats=[1],channels=["logit"],stage="run",
        tag="windows_lock",subjects=2,stimuli=4,seed=12,grid="shared",
        domain="all",standardized=False,precision="float64",
        device="cpu",beta=None,zdim=None,pca=0,
        blocks="U,UX",methods="ad_mixed",step=.03,sigma=.05,
        mc=8,stein_samples=32,surrogate_train=100,centers=20,epochs=3,
        chunk=4,zero_tol=1e-8,directional_points=4,
        storage="metrics",taylor=False,taylor_points=2,
        force=False,fail_fast=True,audit=False)
    folder=run.run_case("C1",1,"logit",args)
    assert not (folder/"ad_mixed"/"U.npy").exists()
    assert not (folder/"ad_mixed"/"UX.npy").exists()
    import json
    manifest=json.loads((folder/"ad_mixed"/"run_manifest.json").read_text())
    assert manifest["status"] == "complete"

def test_cr3_domain_masks_raw_units_even_when_standardized():
    """Zero-amount rows must not become 'interior' through z-scoring."""
    import pytest
    from types import SimpleNamespace
    case=SimpleNamespace(model="CR3",
        x_mu=np.array([20.,10.,.5,20.,10.,.5]),
        x_sd=np.array([20.,10.,.2,20.,10.,.2]))
    raw=np.array([[0.,4.,.5,40.,0.,.5],
                  [30.,10.,.5,25.,8.,.5]])
    standard=(raw-case.x_mu)/case.x_sd
    selected,details=run.set_domain(case,standard,"interior",standardized=True)
    assert details["retained_stimuli"]==1
    assert np.allclose(selected[0], standard[1])
    smooth,info=run.set_domain(case,standard,"smooth_probes",standardized=True,
                               amount_floor=1.0,prob_epsilon=0.01)
    smooth_raw=smooth*case.x_sd+case.x_mu
    assert smooth_raw[0,0]>=1.0 and smooth_raw[0,4]>=1.0
    assert np.allclose(smooth_raw[1],raw[1])
    assert info["changed_rows"]==1
    assert info["changed_elements_by_x_column"]==[1,0,0,0,1,0]
    assert run.strict_x_interior(case,smooth_raw).all()
    all_x,all_info=run.set_domain(case,standard,"all",standardized=True)
    assert np.allclose(all_x,standard)
    with pytest.raises(RuntimeError,match="smooth_probes"):
        run.set_domain(case,standard[:1],"interior",standardized=True)
