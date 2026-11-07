"""CR3 experiment 05: extend the step-04 ZX Hessian matrix estimator.

READ ONLY the step-04 frozen six Y / original Z,X / analytic Hessian arrays.
No checkpoint loading, NN gradients, NN Hessians, or regeneration of Y.

Example:
 python math/05_cr3_zx_hessian_optimization/run.py --methods tensor krr spline
 python math/05_cr3_zx_hessian_optimization/run.py --methods tensor --preset full
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from tensor import TensorExperiment
from patterns import PARTS,all_blocks
from four_d import run_krr,run_splines
from rbf import make_rbf_experiment

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CONDITION="beta5em05_lambda0_ctx0p25to0p8_L4"
CHANNELS=("oracle_choice","oracle_logit","oracle_probability",
          "nn_choice","nn_logit","nn_probability")
CLASSES={"oracle_choice":"probability","nn_choice":"probability",
         "oracle_logit":"logit","nn_logit":"logit",
         "oracle_probability":"probability","nn_probability":"probability"}
LABELS=("z1","z2","z3","z4","gain_A","loss_A","prob_A",
        "gain_B","loss_B","prob_B")


def sha256(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for chunk in iter(lambda:f.read(1048576),b""):h.update(chunk)
    return h.hexdigest()


def as_builtin(obj):
    if isinstance(obj,np.ndarray):return as_builtin(obj.tolist())
    if isinstance(obj,(np.floating,float)):
        f=float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj,(np.integer,int)):return int(obj)
    if isinstance(obj,(np.bool_,bool)):return bool(obj)
    if isinstance(obj,dict):return {str(k):as_builtin(v) for k,v in obj.items()}
    if isinstance(obj,(tuple,list)):return [as_builtin(x) for x in obj]
    return obj


def load_step04(directory,seed):
    """Read exact stored 04 outputs; do not regenerate stochastic Y."""
    folder=Path(directory)
    required=(folder/"all_Y_and_mapping.npz",
              folder/"all_matrices.npz",folder/"summary.json")
    for path in required:
        if not path.is_file():raise FileNotFoundError(
            f"Missing step-04 result {path}. Run math/04_cr3_zx_hessian_threeway/run.py first.")
    summary=json.loads(required[2].read_text(encoding="utf-8"))
    if not summary.get("no_nn_hessian"):
        raise ValueError("Expected step-04 no-neural-Hessian provenance")
    with np.load(required[0],allow_pickle=False) as f:
        needed=set(CHANNELS)|{"Z_all_trial","X_fixed","A","b","T_hat_AZ_plus_b"}
        if not needed.issubset(f.files):
            raise ValueError("Incomplete step-04 all_Y_and_mapping.npz")
        Z=np.array(f["Z_all_trial"],dtype=np.float64)
        X=np.array(f["X_fixed"],dtype=np.float64)
        Y=np.stack([f[k] for k in CHANNELS],axis=-1)
        A=np.array(f["A"],dtype=float)
        b=np.array(f["b"],dtype=float)
        T_hat=np.array(f["T_hat_AZ_plus_b"],dtype=float)
    with np.load(required[1],allow_pickle=False) as f:
        train=np.asarray(f["sieve_train_ids"],dtype=int) if "sieve_train_ids" in f.files else None
        held=np.asarray(f["sieve_test_ids"],dtype=int) if "sieve_test_ids" in f.files else None
        oracle={}
        for name in ("logit","probability"):
            oracle[name]={
                field:np.array(f[f"analytic_{name}__{field}"],dtype=float)
                for field in ("conditional_mean","fraction_defined",
                              "conditional_rms","conditional_std")}
        old={k:np.array(f["fitted_"+k+"__conditional_mean"],dtype=float)
             for k in CHANNELS}
    if train is None or held is None:
        raise ValueError("Step-04 must save sieve_train_ids and sieve_test_ids")
    n,m=Y.shape[:2]
    if Z.shape!=(n,4) or X.shape!=(m,6) or Y.shape[2]!=len(CHANNELS):
        raise ValueError("Expected aligned 4D Z / 6D X / six channels")
    if A.shape!=(4,4) or b.shape!=(4,) or T_hat.shape!=(n,4):
        raise ValueError("Invalid step-04 A,b,T_hat shapes")
    if not np.allclose(T_hat,Z@A.T+b,rtol=1e-7,atol=1e-7):
        raise ValueError("Step-04 T_hat is not A@Z+b")
    if not np.isfinite(Y).all() or not np.isfinite(Z).all():
        raise ValueError("Nonfinite Z or Y")
    if not np.array_equal(np.sort(np.r_[train,held]),np.arange(n)):
        raise ValueError("Step-04 original sieve train/test indices invalid")
    # Original 04 trained on 80%. Split the untouched 20% into equal
    # validation and final test groups; no Hessian oracle tunes models.
    order=np.random.default_rng(seed+71).permutation(held)
    mid=len(order)//2
    val,test=order[:mid],order[mid:]
    if not len(val) or not len(test):raise ValueError("Empty validation/test")
    inputs=dict(Z=Z,X=X,Y=Y,oracle=oracle,old=old,train=train,val=val,test=test,
                A=A,b=b,T_hat=T_hat,source_summary=summary,
                checksums={p.name:sha256(p) for p in required})
    return inputs


def mean_block_scores(H,oracle,channel):
    ref=oracle[CLASSES[channel]]
    return all_blocks(ref["conditional_mean"],H,ref["fraction_defined"])


def plot_triple(path,reference,old,current,name,block):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sl={"ZZ":(slice(0,4),slice(0,4)),
        "ZX":(slice(0,4),slice(4,10)),
        "XX":(slice(4,10),slice(4,10))}[block]
    matrices=[m[sl] for m in (reference,old,current)]
    limit=max(1e-10,max(float(np.nanmax(abs(x))) for x in matrices))
    fig,axes=plt.subplots(1,3,figsize=(13,4.1),constrained_layout=True)
    for ax,M,label in zip(axes,matrices,("CR3 analytic","Step04 polynomial","Step05 improved")):
        img=ax.imshow(np.ma.masked_invalid(M),vmin=-limit,vmax=limit,cmap="coolwarm")
        ax.set_title(label)
        ax.set_xticks(range(M.shape[1]),LABELS[sl[1]],rotation=55,ha="right",fontsize=7)
        ax.set_yticks(range(M.shape[0]),LABELS[sl[0]],fontsize=7)
    fig.colorbar(img,ax=axes,shrink=.7)
    fig.suptitle(name+" · "+block)
    fig.savefig(path,dpi=145)
    plt.close(fig)


def csv_rows(path,records):
    if not records:return
    keys=list(dict.fromkeys(k for r in records for k in r))
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader()
        for row in records:
            w.writerow({k:as_builtin(v) for k,v in row.items()})


def run_tensors(data,args,output):
    Z,X,Y=data["Z"],data["X"],data["Y"]
    ridges_z=args.ridge_z if args.ridge_z else ([.01,1e-4,1e-6] if args.preset=="quick" else [.01,1e-4,1e-6,1e-8])
    ridges_x=args.ridge_x if args.ridge_x else ([.01,1e-4,1e-6] if args.preset=="quick" else [.01,1e-4,1e-6,1e-8])
    degrees_z=args.degree_z if args.degree_z else ([4] if args.preset=="quick" else [3,4,5])
    degrees_x=args.degree_x if args.degree_x else ([3,4] if args.preset=="quick" else [2,3,4])
    bases=args.basis if args.basis else ["monomial","chebyshev"]
    combos=list(itertools.product(bases,degrees_z,degrees_x))
    if len(combos)*len(ridges_z)*len(ridges_x)>args.max_grid:
        raise ValueError("Too many grid candidates. Narrow --basis/--degree-*/--ridge-* or increase --max-grid.")
    print(f"Tensor grid: {len(combos)} basis/degree designs × "
          f"{len(ridges_z)*len(ridges_x)} ridge settings = "
          f"{len(combos)*len(ridges_z)*len(ridges_x)} fits for ALL six Y",flush=True)
    grid_rows=[]
    selected={basis:{} for basis in bases}
    count=0
    for basis,dz,dx in combos:
        print(f"Design {basis}: degree_Z={dz}, degree_X={dx}",flush=True)
        engine=TensorExperiment(Z,X,Y,data["train"],data["val"],data["test"],
                                degree_z=dz,degree_x=dx,basis=basis)
        for rz,rx in itertools.product(ridges_z,ridges_x):
            count+=1
            coefficients=engine.fit(rz,rx)
            valid=engine.metrics(coefficients,data["val"])["rmse"]
            heldout=engine.metrics(coefficients,data["test"])["rmse"]
            H,_=engine.mean_hessian(coefficients)
            for c,name in enumerate(CHANNELS):
                matrices=mean_block_scores(H[:,:,c],data["oracle"],name)
                common=dict(method=basis,degree_z=dz,degree_x=dx,
                            ridge_z=float(rz),ridge_x=float(rx),channel=name,
                            val_rmse=float(valid[c]),test_rmse=float(heldout[c]),
                            n_features=engine.feature_count)
                for part,metrics in matrices.items():
                    grid_rows.append({**common,"block":part,**metrics})
                previous=selected[basis].get(name)
                # The analytic Hessian is NOT consulted here: choose only by
                # validation-response RMSE, never oracle curvature scores.
                if previous is None or float(valid[c])<previous["val_rmse"]:
                    selected[basis][name]={
                        **common,"hessian":H[:,:,c].copy(),
                        "coefficients":coefficients[:,:,c].copy(),
                        "pattern_scores":matrices}
        print(f" finished {count}/{len(combos)*len(ridges_z)*len(ridges_x)} settings",flush=True)
        del engine
    csv_rows(output/"tensor_grid_pattern_metrics.csv",grid_rows)
    output_arrays={}
    summary={}
    for basis,items in selected.items():
        summary[basis]={}
        for name,best in items.items():
            mat=best["hessian"]
            output_arrays[f"{basis}__{name}__mean_hessian"]=mat
            output_arrays[f"{basis}__{name}__coefficients"]=best["coefficients"]
            report={k:v for k,v in best.items() if k not in ("hessian","coefficients")}
            summary[basis][name]=report
            for block in ("ZZ","ZX","XX"):
                plot_triple(output/f"{basis}__{name}__{block}.png",
                            data["oracle"][CLASSES[name]]["conditional_mean"],
                            data["old"][name],mat,basis+"/"+name,block) if not args.no_plots else None
            print(f" selected {basis}/{name}: degrees ({best['degree_z']},{best['degree_x']}), "
                  f"lambda ({best['ridge_z']:g},{best['ridge_x']:g}), "
                  f"ZZ cosine {best['pattern_scores']['ZZ']['cosine']}, "
                  f"ZX cosine {best['pattern_scores']['ZX']['cosine']}, "
                  f"XX cosine {best['pattern_scores']['XX']['cosine']}",flush=True)
    return summary,output_arrays,len(combos)*len(ridges_z)*len(ridges_x)


def run_rbfs(data,args,output):
    """RBF full 10D Hessian, same six Y, same splits, same oracle evaluator."""
    lengths_z=args.rbf_ell_z or ([.75,1.5] if args.preset=="quick" else [.5,1.,2.])
    lengths_x=args.rbf_ell_x or ([.75,1.5] if args.preset=="quick" else [.5,1.,2.])
    ridges=args.rbf_ridges or ([.01,1e-4,1e-6] if args.preset=="quick" else [.01,1e-4,1e-6,1e-8])
    number=len(lengths_z)*len(lengths_x)*len(ridges)**2
    if number>args.max_grid:
        raise ValueError(f"RBF grid has {number} candidates; narrow parameters or raise --max-grid")
    rows=[]
    selected={}
    Z,X,Y=data["Z"],data["X"],data["Y"]
    count=0
    for ell_z,ell_x in itertools.product(lengths_z,lengths_x):
        print(f"RBF full ZX: ell_Z={ell_z:g}, ell_X={ell_x:g}",flush=True)
        engine=make_rbf_experiment(
            Z,X,Y,data["train"],data["val"],data["test"],
            landmarks_z=args.rbf_landmarks_z,
            landmarks_x=args.rbf_landmarks_x,
            ell_z=ell_z,ell_x=ell_x,seed=args.seed)
        for rz,rx in itertools.product(ridges,ridges):
            count+=1
            coeff=engine.fit(rz,rx)
            v=engine.metrics(coeff,data["val"])["rmse"]
            test=engine.metrics(coeff,data["test"])["rmse"]
            H,_=engine.mean_hessian(coeff)
            for j,name in enumerate(CHANNELS):
                blocks=mean_block_scores(H[:,:,j],data["oracle"],name)
                fields=dict(method="rbf",ell_z=float(ell_z),ell_x=float(ell_x),
                            n_landmarks_z=int(args.rbf_landmarks_z),
                            n_landmarks_x=int(args.rbf_landmarks_x),
                            ridge_z=float(rz),ridge_x=float(rx),
                            channel=name,val_rmse=float(v[j]),
                            test_rmse=float(test[j]),n_features=engine.feature_count)
                for block,metrics in blocks.items():
                    rows.append({**fields,"block":block,**metrics})
                previous=selected.get(name)
                if previous is None or float(v[j])<previous["val_rmse"]:
                    selected[name]={**fields,
                                    "hessian":H[:,:,j].copy(),
                                    "coefficients":coeff[:,:,j].copy(),
                                    "pattern_scores":blocks}
        print(f" completed {count}/{number} RBF candidates",flush=True)
        del engine
    csv_rows(output/"rbf_grid_pattern_metrics.csv",rows)
    arrays={};summary={}
    for name,best in selected.items():
        matrix=best["hessian"]
        arrays[f"rbf__{name}__mean_hessian"]=matrix
        arrays[f"rbf__{name}__coefficients"]=best["coefficients"]
        summary[name]={k:v for k,v in best.items()
                       if k not in ("hessian","coefficients")}
        if not args.no_plots:
            for block in ("ZZ","ZX","XX"):
                plot_triple(output/f"rbf__{name}__{block}.png",
                            data["oracle"][CLASSES[name]]["conditional_mean"],
                            data["old"][name],matrix,"rbf/"+name,block)
        print(f" selected RBF/{name}: ell_Z={best['ell_z']:g}, "
              f"ell_X={best['ell_x']:g}, ridge=({best['ridge_z']:g},{best['ridge_x']:g}) "
              f"ZZ cosine {best['pattern_scores']['ZZ']['cosine']}, "
              f"ZX cosine {best['pattern_scores']['ZX']['cosine']}, "
              f"XX cosine {best['pattern_scores']['XX']['cosine']}",flush=True)
    return summary,arrays,number


def run_zz_methods(data,args,output,methods):
    results={};arrays={}
    Z,Y=data["Z"],data["Y"]
    evals=[]
    if "krr" in methods:
        evals.append(("krr",lambda:run_krr(
            Z,Y,data["train"],data["val"],data["test"],
            rank=args.krr_rank,seed=args.seed)))
    if "spline" in methods:
        basis_values=[5] if args.preset=="quick" else [5,6]
        basis_values=args.spline_bases if args.spline_bases else basis_values
        for n in basis_values:
            evals.append((f"spline_n{n}",lambda n=n:run_splines(
                Z,Y,data["train"],data["val"],data["test"],n_basis=n)))
    for method,func in evals:
        print(f"Fitting 4D ZZ-only method: {method}",flush=True)
        H,reports=func()
        if H.shape!=(4,4,len(CHANNELS)):
            raise ValueError("ZZ-only estimator did not return 4x4 matrices")
        results[method]={}
        for c,name in enumerate(CHANNELS):
            ref=data["oracle"][CLASSES[name]]
            refH=ref["conditional_mean"]
            mask=ref["fraction_defined"]
            # Build 10x10 arrays only for the ZZ pattern-scoring interface;
            # never fabricate estimates for ZX or XX.
            padded=np.full((10,10),np.nan);padded[:4,:4]=H[:,:,c]
            metric=all_blocks(refH,padded,mask)
            zz_only={k:metric[k] for k in ("ZZ","ZZ_diag","ZZ_offdiag")}
            results[method][name]=dict(**reports[c],pattern_scores=zz_only)
            arrays[f"{method}__{name}__ZZ_mean"]=H[:,:,c]
            print(f" {method}/{name}: validation RMSE {reports[c]['validation_rmse']:.5g} "
                  f"ZZ cosine {metric['ZZ']['cosine']}",flush=True)
    return results,arrays


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--condition",default=CONDITION)
    p.add_argument("--repeat",type=int,default=1)
    p.add_argument("--source-dir",type=Path,
                   help="Previously completed step-04 repeat directory")
    p.add_argument("--output",type=Path)
    p.add_argument("--preset",choices=("quick","full"),default="quick")
    p.add_argument("--methods",nargs="+",choices=("tensor","rbf","krr","spline"),
                   default=["tensor","rbf","krr","spline"])
    p.add_argument("--basis",nargs="+",choices=("monomial","chebyshev"))
    p.add_argument("--degree-z",nargs="+",type=int)
    p.add_argument("--degree-x",nargs="+",type=int)
    p.add_argument("--ridge-z",nargs="+",type=float)
    p.add_argument("--ridge-x",nargs="+",type=float)
    p.add_argument("--rbf-landmarks-z",type=int,default=64)
    p.add_argument("--rbf-landmarks-x",type=int,default=64)
    p.add_argument("--rbf-ell-z",nargs="+",type=float)
    p.add_argument("--rbf-ell-x",nargs="+",type=float)
    p.add_argument("--rbf-ridges",nargs="+",type=float)
    p.add_argument("--krr-rank",type=int,default=128)
    p.add_argument("--spline-bases",nargs="+",type=int)
    p.add_argument("--max-grid",type=int,default=400)
    p.add_argument("--seed",type=int,default=20261106)
    p.add_argument("--no-plots",action="store_true")
    p.add_argument("--verify-step04-baseline",action="store_true",
                   help="Reproduce step04 old degree4/3 ridge .01 mean matrices (strict test)")
    args=p.parse_args()
    if (args.repeat<1 or args.krr_rank<8 or args.max_grid<1 or
        args.rbf_landmarks_z<4 or args.rbf_landmarks_x<4):
        p.error("Invalid repeat / krr rank / max grid")
    if args.source_dir is None:
        args.source_dir=ROOT/"outputs/math/cr3_zx_hessian_threeway"/args.condition/f"repeat{args.repeat:02d}"
    if args.output is None:
        args.output=(ROOT/"outputs/math/05_cr3_zx_hessian_optimization"/args.condition/
                     f"repeat{args.repeat:02d}"/f"{args.preset}__{'_'.join(args.methods)}")
    data=load_step04(args.source_dir,args.seed)
    args.output.mkdir(parents=True,exist_ok=True)
    if args.verify_step04_baseline:
        print("Strict reproduction of step04 TensorPolynomial (degree 4/3, lambda .01/.01)...",flush=True)
        m=TensorExperiment(data["Z"],data["X"],data["Y"],data["train"],
                           data["val"],data["test"],degree_z=4,degree_x=3,basis="monomial")
        oldH,_=m.mean_hessian(m.fit(.01,.01))
        maxdiff=max(float(np.nanmax(np.abs(oldH[:,:,c]-data["old"][name])))
                    for c,name in enumerate(CHANNELS))
        print("  max old/new Hessian difference:",maxdiff,flush=True)
        if maxdiff>1e-8:
            raise AssertionError(f"Failed step04 baseline reproduction: {maxdiff}")
    report=dict(
        study="05: improved and extended ZX Hessian estimators on frozen step04 six Y",
        source_dir=str(args.source_dir),
        source_checksums=data["checksums"],methods=args.methods,preset=args.preset,
        n_subjects=len(data["Z"]),n_stimuli=len(data["X"]),
        channels=list(CHANNELS),coordinate_system=list(LABELS),
        train_count=len(data["train"]),validation_count=len(data["val"]),
        test_count=len(data["test"]),seed=args.seed,
        parameter_selection="min validation Y RMSE ONLY; analytic Hessian is NEVER used for hyperparameter selection",
        original_step04_sieve_train_ids_reused=True,
        unchanged_Y_Z_X_and_oracle=True,
        no_nn_hessian=True,
        interpretation=[
            "The frozen original Z, X, 6 Y and analytic CR3 H come directly from step04.",
            "Original predictor study included 80% train and 20% test. The old 20% is split here into validation/test; all regressors use the exact original 80% train subjects.",
            "10D tensor methods estimate full ZZ, ZX, XX; KRR and B-spline model only subject-averaged Y and provide ZZ exclusively.",
            "Hard NN choice Hessian is not smooth; its discrepancy to CR3 probability Hessian is an external diagnostic only.",
            "Oracle Y and NN Y are not necessarily the same function. Oracle-H discrepancy does not isolate regression error for NN-Y.",
            "At X=0, some amount derivatives are undefined; conditional oracle means use the old domain-specific masks.",
            "The source 04 affine Z->T map may have used all subjects; this is not an out-of-sample map-recovery test.",
            "A small prediction RMSE does not guarantee an accurate Hessian. All candidate curvature scores are recorded, but never used to choose a model.",
        ],
        configs=dict(basis=args.basis,degree_z=args.degree_z,degree_x=args.degree_x,
                     ridge_z=args.ridge_z,ridge_x=args.ridge_x,
                     krr_rank=args.krr_rank,spline_bases=args.spline_bases,
                     rbf_landmarks_z=args.rbf_landmarks_z,
                     rbf_landmarks_x=args.rbf_landmarks_x,
                     rbf_ell_z=args.rbf_ell_z,rbf_ell_x=args.rbf_ell_x,
                     rbf_ridges=args.rbf_ridges),
        results={},
    )
    arrays={}
    if "tensor" in args.methods:
        report["results"]["tensor"],a,number=run_tensors(data,args,args.output)
        report["tensor_designs_times_ridge_pairs"]=number
        arrays.update(a)
    if "rbf" in args.methods:
        report["results"]["rbf"],rbf_arrays,rbf_number=run_rbfs(data,args,args.output)
        report["rbf_bandwidth_ridge_candidates"]=rbf_number
        arrays.update(rbf_arrays)
    other,rest=run_zz_methods(data,args,args.output,args.methods)
    report["results"]["zz_only"]=other
    arrays.update(rest)
    for k in ("logit","probability"):
        arrays[f"analytic_{k}__mean"]=data["oracle"][k]["conditional_mean"]
        arrays[f"analytic_{k}__valid_fraction"]=data["oracle"][k]["fraction_defined"]
    for k in CHANNELS:arrays[f"step04_{k}__mean"]=data["old"][k]
    arrays["train_ids"]=data["train"];arrays["val_ids"]=data["val"];arrays["test_ids"]=data["test"]
    np.savez_compressed(args.output/"selected_hessians.npz",**arrays)
    (args.output/"summary.json").write_text(json.dumps(as_builtin(report),indent=2,
                                             allow_nan=False),encoding="utf-8")
    print("Saved new step05 results:",args.output,flush=True)


if __name__=="__main__":
    main()
