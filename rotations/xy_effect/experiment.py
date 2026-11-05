"""A single self-contained rotation experiment package. No method registry/batching.

Imports only this folder's own api.py/data.py/model and NP's frozen
S1(X,Y) + S5(all-observed-trial Z). The simulated true parameters are only
accessed after R and all-subject U have been written to disk.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from .api import RotationInput, optimize_objective, validate_result
from .data import aligned_observations, evaluate_after_freeze

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SUITE = HERE.name


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=f"Standalone {SUITE} rotation experiment")
    p.add_argument("--model",choices=("C1","C2","C3","CR3"),required=True)
    p.add_argument("--condition",required=True,help="NP training tag, e.g. beta0_lambda0_ctx0p25to0p8_L8")
    p.add_argument("--repeat",type=int,default=1)
    p.add_argument("--representation",choices=("pca","full"),default="pca")
    p.add_argument("--rotation-dim",type=int,default=None)
    p.add_argument("--fit-max-subjects",type=int,default=2500)
    p.add_argument("--whiten-floor",type=float,default=1e-5)
    p.add_argument("--seed",type=int,default=31)
    p.add_argument("--starts",type=int,default=None,help="Override local starts, if configured")
    p.add_argument("--maxiter",type=int,default=None,help="Override local maxiter, if configured")
    p.add_argument("--set",action="append",default=[],help="Override THIS suite config: --set ridge=0.001")
    p.add_argument("--no-evaluate",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--overwrite",action="store_true")
    args=p.parse_args(argv)
    if args.repeat<1 or args.fit_max_subjects<1 or args.whiten_floor<=0:
        p.error("repeat, fit-max-subjects and whiten-floor must be positive")
    plugin=importlib.import_module(__package__)
    config_path=HERE/"config.json"
    config=json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config,dict):
        p.error("Local config.json must be a JSON object")
    # Local DEFAULTS are the only hyperparameter names allowed by this suite.
    defaults=getattr(plugin,"DEFAULTS",{})
    if not isinstance(defaults,dict) or set(config)-set(defaults):
        p.error(f"Local config keys must be declared in DEFAULTS: {sorted(defaults)}")
    config={**defaults,**config}
    if "seed" in config:
        config["seed"]=args.seed
    for key in ("starts","maxiter"):
        value=getattr(args,key)
        if value is not None:
            if key not in config:
                p.error(f"This suite has no {key} parameter")
            config[key]=value
    for expression in args.set:
        key,sep,value=expression.partition("=")
        if not sep or key not in config:
            p.error(f"Unknown suite config {key!r}; allowed: {sorted(config)}")
        try:
            config[key]=json.loads(value)
        except json.JSONDecodeError:
            p.error(f"Value for {key} must be valid JSON")
    return args,config,plugin


def load_source(args):
    from modlearn_rot.generators.dataset import load_dataset
    root=ROOT/"outputs"/"choice"/args.model
    z_path=root/"s4_evaluate"/args.condition/f"repeat{args.repeat:02d}"/"s4_evaluate__latent.npz"
    dataset_path=root/"s1_generate"/"s1_generate__dataset.npz"
    if not z_path.is_file():
        raise FileNotFoundError(f"Run np/s5_extract_z.py before this suite: {z_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Run np/s1_generate.py before this suite: {dataset_path}")
    data_sha=sha256(dataset_path)
    with np.load(z_path,allow_pickle=False) as f:
        if str(f["context_mode"])!="all_observed_trials":
            raise RuntimeError("Z must come from S5 all-trial extraction")
        if str(f["dataset_sha256"])!=data_sha:
            raise RuntimeError("S1 dataset changed since S5 extraction")
        z=np.asarray(f["z"],dtype=float)
        tr=np.asarray(f["train_ids"],dtype=int)
        val=np.asarray(f["validation_ids"],dtype=int)
        te=np.asarray(f["test_ids"],dtype=int)
        ckpt_sha=str(f["checkpoint_sha256"])
    if z.ndim!=2 or not np.isfinite(z).all():
        raise ValueError("Invalid latent Z")
    if not np.array_equal(np.sort(np.r_[tr,val,te]),np.arange(len(z))):
        raise ValueError("Subject splits are invalid or overlapping")
    native_dim=z.shape[1]
    dim=args.rotation_dim or (
        native_dim if args.representation=="full"
        else {"C1":4,"C2":4,"C3":6,"CR3":4}[args.model])
    if dim<2 or dim>native_dim or (args.representation=="full" and dim!=native_dim):
        raise ValueError("Invalid rotation dimension")
    pca=PCA(n_components=dim,svd_solver="full").fit(z[tr])
    W=pca.components_/np.sqrt(np.maximum(pca.explained_variance_,args.whiten_floor))[:,None]
    q=(z-pca.mean_)@W.T
    chosen=np.sort(np.random.default_rng(args.seed).choice(
        tr,size=min(len(tr),args.fit_max_subjects),replace=False))
    # Dataset contains theta for later TEST diagnostics, but only X,Y are
    # passed to optimization. Never pass the dataset or theta to suite.fit().
    dataset=load_dataset(dataset_path)
    X,Y,mask=aligned_observations(dataset)
    if len(X)!=len(z):
        raise ValueError("S1 and S5 subject counts disagree")
    fitted=RotationInput(np.array(q[chosen],copy=True),
                         np.array(X[chosen],copy=True),
                         np.array(Y[chosen],copy=True),
                         np.array(mask[chosen],copy=True))
    return dict(data=fitted,z=z,tr=tr,val=val,te=te,chosen=chosen,
                W=W,pca=pca,root=root,dataset=dataset,dim=dim,native_dim=native_dim,
                latent_sha=sha256(z_path),dataset_sha=data_sha,checkpoint_sha=ckpt_sha)


def method_code_sha():
    """Hash only this suite's own code (including helper modules)."""
    digest=hashlib.sha256()
    for path in sorted(HERE.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(HERE).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def execute(args, config, plugin, source=None):
    source=load_source(args) if source is None else source
    fit=source["data"]
    meta=dict(suite=SUITE,suite_version=str(getattr(plugin,"VERSION","1")),
              suite_source_sha256=method_code_sha(),model=args.model,
              condition=args.condition,repeat=args.repeat,
              representation=args.representation,rotation_dim=source["dim"],
              native_latent_dim=source["native_dim"],
              fit_max_subjects=args.fit_max_subjects,
              whiten_floor=args.whiten_floor,seed=args.seed,
              configuration=config,
              dataset_sha256=source["dataset_sha"],latent_sha256=source["latent_sha"],
              checkpoint_sha256=source["checkpoint_sha"],
              fit_ids_sha256=hashlib.sha256(source["chosen"].tobytes()).hexdigest(),
              no_evaluate=bool(args.no_evaluate))
    fingerprint=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()).hexdigest()[:16]
    folder=(source["root"]/"rotations"/SUITE/args.condition/
            f"repeat{args.repeat:02d}"/f"run_{fingerprint}")
    if (folder/"rotation__complete.json").is_file() and not args.overwrite:
        print("[SKIP] Completed exact suite:",folder)
        return folder
    if folder.is_dir() and any(folder.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Incomplete suite run: {folder}. Inspect or --overwrite")
    if args.dry_run:
        print("[PLAN]",SUITE,meta,folder)
        return folder
    if callable(getattr(plugin,"fit",None)):
        result=plugin.fit(fit,config)
    else:
        result=optimize_objective(fit,plugin.objective,config)
    result=validate_result(result,fit.k)
    R=result.R
    A=R@source["W"]
    offset=-A@source["pca"].mean_
    U=source["z"]@A.T+offset
    if not np.allclose(U[source["chosen"]],fit.rotated(R),atol=1e-6):
        raise RuntimeError("U is not consistent with R, PCA and original Z")
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/"rotation__matrix.npz",
                        R=R,A=A,offset=offset,
                        pca_components=source["pca"].components_,
                        pca_eigenvalues=source["pca"].explained_variance_,
                        pca_mean=source["pca"].mean_,fit_ids=source["chosen"],
                        source_z_sha256=np.array(source["latent_sha"]),
                        source_dataset_sha256=np.array(source["dataset_sha"]),
                        suite=np.array(SUITE))
    np.savez_compressed(folder/"rotation__U.npz",
                        U=U,train_ids=source["tr"],validation_ids=source["val"],
                        test_ids=source["te"])
    if result.extra_arrays:
        np.savez_compressed(folder/"rotation__suite_arrays.npz",
                            **{key:np.asarray(value) for key,value in result.extra_arrays.items()})
    info={**meta,"fingerprint":fingerprint,"fit_subject_count":fit.n_subjects,
          "optimizer_result":result.info,"state":"fit_saved"}
    (folder/"rotation__fit.json").write_text(json.dumps(info,indent=2,allow_nan=False),encoding="utf-8")
    # True parameter values are ONLY read after all FIT artifacts are on disk.
    if not args.no_evaluate:
        from modrot3.analysis import param_names
        labels=param_names(args.model)
        report=evaluate_after_freeze(source["dataset"].parameters,U,source["te"],
                                      folder,SUITE,labels)
        print(f"[TEST] {SUITE} matched mean r²={report['matched_mean_r2']:.4f}")
    (folder/"rotation__complete.json").write_text(
        json.dumps(dict(fingerprint=fingerprint,evaluated=not args.no_evaluate),indent=2),
        encoding="utf-8")
    print("[DONE]",SUITE,folder)
    return folder


def main(argv=None):
    args,config,plugin=parse_args(argv)
    return execute(args,config,plugin)
