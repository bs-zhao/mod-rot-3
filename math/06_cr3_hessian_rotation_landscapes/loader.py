"""Read the completed step-04 experiment without modifying any input.

Accept its original repeat01 directory OR an archived repeat01 ZIP, including
the versions previously uploaded for analysis. Refit OLS A_star on the SAME
original map_fit_ids and verify agreement with the stored A, b.
"""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from rotation import fit_affine

CHANNELS=("oracle_choice","oracle_logit","oracle_probability",
          "nn_choice","nn_logit","nn_probability")
TARGETS=("choice","logit","probability")
ORACLE_TARGET={"choice":"probability","logit":"logit","probability":"probability"}


def _file_bytes(source,directory_name):
    key=directory_name
    if isinstance(source,zipfile.ZipFile):
        matches=[n for n in source.namelist() if n.endswith("/"+key)]
        if len(matches)!=1:
            raise ValueError(f"Expected ONE {key} in ZIP; found {len(matches)}")
        return source.read(matches[0])
    p=Path(source)/key
    if not p.is_file():raise FileNotFoundError(f"Step04 result missing: {p}")
    return p.read_bytes()


def load_source(source):
    """Read exactly the original 04 saved Z, X, Y, CR3 HZ and OLS fit."""
    src=Path(source)
    if not src.exists():raise FileNotFoundError(f"Source not found: {src}")
    with zipfile.ZipFile(src,"r") if src.is_file() else _NoopZip() as handle:
        base=handle if src.is_file() else src
        raw={key:_file_bytes(base,key)
             for key in ("all_Y_and_mapping.npz","all_matrices.npz","summary.json")}
    summary=json.loads(raw["summary.json"].decode("utf-8"))
    if summary.get("no_nn_hessian") is not True:
        raise ValueError("Expected step04 frozen NN output / no NN differentiation")
    with np.load(io.BytesIO(raw["all_Y_and_mapping.npz"]),allow_pickle=False) as f:
        keys=set(CHANNELS)|{"Z_all_trial","T_true","X_fixed","A","b"}
        if not keys.issubset(f.files):
            raise ValueError(f"Missing source data keys: {sorted(keys-set(f.files))}")
        Z=np.array(f["Z_all_trial"],dtype=float)
        T=np.array(f["T_true"],dtype=float)
        X=np.array(f["X_fixed"],dtype=float)
        Y=np.stack([f[key] for key in CHANNELS],axis=-1).astype(np.float64)
        stored_A=np.array(f["A"],dtype=float)
        stored_b=np.array(f["b"],dtype=float)
    with np.load(io.BytesIO(raw["all_matrices.npz"]),allow_pickle=False) as m:
        fitted=m.files
        if "map_fit_ids" not in fitted or "sieve_train_ids" not in fitted or "sieve_test_ids" not in fitted:
            raise ValueError("Step04 must include mapping/sieve subject IDs")
        map_fit=np.asarray(m["map_fit_ids"],dtype=int)
        train=np.asarray(m["sieve_train_ids"],dtype=int)
        held=np.asarray(m["sieve_test_ids"],dtype=int)
        oracle={}
        for key in ("logit","probability"):
            oracle[key]={
                "mean_ZX":np.asarray(m[f"analytic_{key}__conditional_mean"],dtype=float),
                "fraction_defined":np.asarray(m[f"analytic_{key}__fraction_defined"],dtype=float)
            }
    if (Z.ndim!=2 or Z.shape[1]!=4 or T.shape!=Z.shape or
        X.ndim!=2 or X.shape[1]!=6 or Y.shape!=(len(Z),len(X),6)):
        raise ValueError("Inconsistent 04 data dimensions")
    if not np.array_equal(np.sort(np.r_[train,held]),np.arange(len(Z))):
        raise ValueError("04 sieve train + heldout must partition all subjects")
    if not np.isfinite(Y).all() or not np.isfinite(Z).all() or not np.isfinite(X).all():
        raise ValueError("Nonfinite Z/X/Y")
    A_star,b=fit_affine(Z,T,map_fit)
    if not (np.allclose(A_star,stored_A,atol=1e-8,rtol=1e-8) and
            np.allclose(b,stored_b,atol=1e-8,rtol=1e-8)):
        raise ValueError("Refitted A_star does not match step04 stored OLS")
    pred=Z@A_star.T+b
    err=T-pred
    r2=1-np.sum(err*err,axis=0)/np.sum((T-T.mean(0))**2,axis=0)
    return dict(Z=Z,T=T,X=X,Y=Y,A_star=A_star,b=b,
                oracle=oracle,train=train,heldout=held,map_fit_ids=map_fit,
                ols_r2=r2,source_summary=summary,
                original_source=str(src),
                sha256={k:hashlib.sha256(v).hexdigest() for k,v in raw.items()})


class _NoopZip:
    """Allow one uniform with-statement for ZIP and directory inputs."""
    def __enter__(self):return None
    def __exit__(self,exc_type,exc,tb):return False
