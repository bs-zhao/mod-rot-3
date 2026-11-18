"""Isolated read-only loader for CR3/4/5 S1, S3 and S5 artifacts.

The true generating theta is intentionally not loaded here. The S1 dataset
is accessed only for the common X stimulus bank and split consistency.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import numpy as np
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[2]
BANK_NAMES={"CR3":"CR3_4P_fixed500.npz","CR4":"CR4_3P_random500.npz",
            "CR5":"CR5_3P_cr3bank500.npz"}


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def source_paths(model,condition,repeat):
    base=ROOT/"outputs"/"choice"/model
    return dict(
        root=base,
        checkpoint=base/"s3_train"/condition/f"repeat{repeat:02d}"/"s3_train__checkpoint.pt",
        latent=base/"s4_evaluate"/condition/f"repeat{repeat:02d}"/"s4_evaluate__latent.npz",
        dataset=base/"s1_generate"/"s1_generate__dataset.npz",
        bank=base/"s0_design"/BANK_NAMES[model])


def load_source(model,condition,repeat,dim,config):
    paths=source_paths(model,condition,repeat)
    for field in ("checkpoint","latent","dataset","bank"):
        if not paths[field].is_file():
            raise FileNotFoundError(f"Missing {field} for {model}: {paths[field]}")
    hashes={p:sha256(paths[p]) for p in ("checkpoint","latent","dataset","bank")}
    with np.load(paths["latent"],allow_pickle=False) as a:
        if (str(a["dataset_sha256"])!=hashes["dataset"] or
            str(a["checkpoint_sha256"])!=hashes["checkpoint"]):
            raise RuntimeError("S5 latent provenance differs from S1 or S3")
        if str(a["context_mode"])!="all_observed_trials":
            raise RuntimeError("Require S5 all_observed_trials posterior means")
        z=np.asarray(a["z"],dtype=np.float64)
        tr=np.asarray(a["train_ids"],dtype=int)
        va=np.asarray(a["validation_ids"],dtype=int)
        te=np.asarray(a["test_ids"],dtype=int)
    if z.ndim!=2 or z.shape[1]<dim:
        raise ValueError("S5 latent width is insufficient")
    if not np.array_equal(np.sort(np.r_[tr,va,te]),np.arange(len(z))):
        raise RuntimeError("S5 train/valid/test subject ids invalid")
    with np.load(paths["dataset"],allow_pickle=False) as a:
        count=np.asarray(a["trial_counts"],dtype=int)
        ids=np.asarray(a["subject_index"],dtype=int)
        if count.shape!=(len(z),) or np.any(count!=500):
            raise ValueError("Each CR3/4/5 subject must have exactly 500 trials")
        if not np.array_equal(ids,np.repeat(np.arange(len(z)),500)):
            raise ValueError("S1 subject ordering differs from S5")
        physical_x=np.asarray(a["x"][:500],dtype=np.float64)
        if not np.array_equal(physical_x,np.asarray(a["x"][-500:])):
            raise ValueError("Shared stimulus bank differs across subjects")
    with np.load(paths["bank"],allow_pickle=False) as a:
        if not np.array_equal(physical_x,np.asarray(a["X"],dtype=np.float64)):
            raise RuntimeError("S1 X does not match its immutable model bank")
        if model in ("CR3","CR5"):
            group_codes=np.asarray(a["type_codes"],dtype=int)
            group_names=list(map(str,a["type_names"]))
            if len(group_codes)!=500 or len(group_names)!=4:
                raise ValueError("Expected CR3/CR5 native four-condition bank")
            if not np.array_equal(np.unique(group_codes),np.arange(4)):
                raise ValueError("CR3/CR5 group code mismatch")
        else:
            group_codes=np.zeros(500,dtype=int)
            group_names=["all"]
    pca=PCA(n_components=dim,svd_solver="full").fit(z[tr])
    # q = PCA-whitened subject coordinates; B reconstructs native Z.
    W=pca.components_/np.sqrt(np.maximum(pca.explained_variance_,
                                          float(config["whiten_floor"])))[:,None]
    B=np.linalg.pinv(W)
    q=(z-pca.mean_)@W.T
    residual=z-pca.mean_-q@B.T
    if not np.allclose(z,pca.mean_+q@B.T+residual,atol=1e-9):
        raise RuntimeError("PCA-whitening roundtrip failed")
    rng=np.random.default_rng(int(config["seed"]))
    nfit=min(len(tr),int(config["fit_subjects"]))
    nval=min(len(va),int(config["valid_subjects"]))
    fit_subjects=np.sort(rng.choice(tr,nfit,replace=False))
    valid_subjects=np.sort(rng.choice(va,nval,replace=False))
    trials=rng.permutation(500)
    nf=min(500,int(config["fit_x"]))
    nv=min(500-nf,int(config["valid_x"]))
    fit_x_ids=np.sort(trials[:nf])
    valid_x_ids=np.sort(trials[nf:nf+nv])
    if min(nfit,nval,nf,nv)<4:
        raise ValueError("Need >=4 subjects and X trial anchors in both FIT/VALID")
    if model in ("CR3","CR5") and config["grouping"]=="balanced":
        for selected in (fit_x_ids,valid_x_ids):
            counts=np.bincount(group_codes[selected],minlength=4)
            if np.any(counts<2):
                raise ValueError("Balanced four-condition objective requires >=2 sampled X per condition; increase --fit-x/--valid-x")
    return {**paths,"model":model,"condition":condition,"repeat":repeat,
            "hashes":hashes,"z":z,"q":q,"W":W,"B":B,"pca_mean":pca.mean_,
            "residual":residual,"train_ids":tr,"validation_ids":va,
            "test_ids":te,"fit_subject_ids":fit_subjects,
            "valid_subject_ids":valid_subjects,"fit_x_ids":fit_x_ids,
            "valid_x_ids":valid_x_ids,"bank_x":physical_x,
            "group_codes":group_codes,"group_names":group_names}
