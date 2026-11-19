"""Read-only S1/S3/S4 Gaussian population fit: NO individual Z used in AD.

For CR3/CR4/CR5, native S5 latent dimension MUST equal the true parameter
dimension (4,3,3). All subjects give ONLY mean and full covariance for
Gaussian integration; supervised oracle reads all z/theta post-freeze.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BANKS={"CR3":"CR3_4P_fixed500.npz",
       "CR4":"CR4_3P_random500.npz",
       "CR5":"CR5_3P_cr3bank500.npz"}
DIMS={"CR3":4,"CR4":3,"CR5":3}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda:file.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def source_paths(model,condition,repeat):
    if model not in DIMS:raise ValueError(f"Unsupported model {model}")
    root=ROOT/"outputs"/"choice"/model
    tag=f"repeat{repeat:02d}"
    return dict(root=root,
        checkpoint=root/"s3_train"/condition/tag/"s3_train__checkpoint.pt",
        latent=root/"s4_evaluate"/condition/tag/"s4_evaluate__latent.npz",
        dataset=root/"s1_generate"/"s1_generate__dataset.npz",
        bank=root/"s0_design"/BANKS[model])


def fit_gaussian(z,dim,eigen_floor=1e-8):
    """For q~N(0,I), z=mu+Bq, with BB^T=full empirical Cov(z).

    B uses a deterministic signed eigenbasis, also defining q for the
    *post-freeze* supervised oracle. No z_i is a derivative integration node.
    """
    z=np.asarray(z,dtype=np.float64)
    if z.ndim!=2 or z.shape[1]!=dim or len(z)<dim+2 or not np.isfinite(z).all():
        raise ValueError("Gaussian requires full-rank native Z across all subjects")
    mu=z.mean(0)
    cov=np.cov(z,rowvar=False,ddof=1)
    vals,vecs=np.linalg.eigh(cov)
    idx=np.argsort(vals)[::-1]
    vals=vals[idx];vecs=vecs[:,idx]
    if np.min(vals)<=float(eigen_floor):
        raise ValueError(f"Degenerate native Z covariance: minimum eigenvalue {min(vals):.4g}")
    for j in range(dim):
        anchor=np.argmax(np.abs(vecs[:,j]))
        if vecs[anchor,j]<0:vecs[:,j]*=-1
    B=vecs*np.sqrt(vals)[None,:]
    W=np.linalg.inv(B)
    q=(z-mu)@W.T
    if not np.allclose(z,mu+q@B.T,rtol=1e-11,atol=1e-10):
        raise RuntimeError("Gaussian affine transform did not reconstruct native z")
    if not np.allclose(np.cov(q,rowvar=False),np.eye(dim),atol=5e-8):
        raise RuntimeError("Whitened q is not unit-covariance")
    return dict(mu=mu,B=B,W=W,covariance=cov,
                eigenvalues=vals,q=q,all_subject_count=len(z))


def load_source(model,condition,repeat,config):
    paths=source_paths(model,condition,repeat)
    for name in ("checkpoint","latent","bank","dataset"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"Missing {name}: {paths[name]}")
    hashes={key:sha256(paths[key])
            for key in ("checkpoint","latent","dataset","bank")}
    with np.load(paths["latent"],allow_pickle=False) as saved:
        if (str(saved["checkpoint_sha256"])!=hashes["checkpoint"] or
            str(saved["dataset_sha256"])!=hashes["dataset"] or
            str(saved["context_mode"])!="all_observed_trials"):
            raise RuntimeError("S5 latent source hash or context mode mismatch")
        z=np.asarray(saved["z"],np.float64)
        ids=np.concatenate([np.asarray(saved[k],int) for k in (
            "train_ids","validation_ids","test_ids")])
        if not np.array_equal(np.sort(ids),np.arange(len(z))):
            raise RuntimeError("NP source split does not cover the full population")
    g=fit_gaussian(z,DIMS[model],float(config["z_cov_eig_floor"]))
    with np.load(paths["bank"],allow_pickle=False) as saved:
        x=np.asarray(saved["X"],dtype=np.float64)
        if model in ("CR3","CR5"):
            groups=np.asarray(saved["type_codes"],int)
            group_names=list(map(str,saved["type_names"]))
        else:
            groups=np.zeros(len(x),dtype=int)
            group_names=["all"]
    if x.shape!=(500,6) or not np.isfinite(x).all() or groups.shape!=(500,):
        raise RuntimeError("CR3/4/5 require an immutable 500 by 6 bank")
    # Strict generator fidelity for the actual CR4 S0 manifest (when available)
    if model=="CR4":
        spec=paths["bank"].parent/"CR4_3P_design.json"
        if spec.is_file():
            design=json.loads(spec.read_text(encoding="utf-8"))
            if design.get("bank_sha256")!=hashes["bank"]:
                raise RuntimeError("CR4 S0 design manifest bank SHA mismatch")
    return {**paths,"model":model,"condition":condition,"repeat":repeat,
            "hashes":hashes,**g,"bank_x":x,"group_codes":groups,
            "group_names":group_names,
            "gaussian_scope":"ALL_subjects_mean_and_full_cov_only"}
