"""Recompute step-06 Hessian sparsity curves from SAVED 10x10 H matrices.

No NP training, Y generation, tensor regression, GPU or second derivative fit.
Accept the original step-06 result DIRECTORY or ZIP, and optionally the
step-04 original result DIRECTORY/ZIP for X-column SD normalization.

Examples (from repo root):
 python math/06_cr3_hessian_rotation_landscapes/rescore_existing.py --input "outputs/math/06_cr3_hessian_rotation_landscapes/.../monomial_u4_x4_ridgeu1e-06_ridgex1e-06"
 python math/06_cr3_hessian_rotation_landscapes/rescore_existing.py --input "D:/Downloads/monomial_u4_x4_ridgeu1e-06_ridgex1e-06.zip"
"""
from __future__ import annotations
import argparse
import io
import json
import zipfile
from pathlib import Path
import numpy as np
from metrics import metrics_for_source,equivariance_error,block_array
from loader import TARGETS,ORACLE_TARGET
from rotation import pair_rotation
from plots import plot_xz_sparsity,DEFAULT_XZ_SPARSITY_METRICS
from attribution import three_way_all
from run import (ALL_BLOCKS,METHODS,csv_write,as_native,
                 calculate_best_angles,plot_index)
from screen import select_sparsity_metrics
from sparsity import metric_catalog

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCENAMES=("analytic","formula_y","nn_y")
TARGETNAMES=("choice","logit","probability")


def read_named(path,name):
    src=Path(path)
    if src.is_dir():
        return (src/name).read_bytes()
    with zipfile.ZipFile(src) as archive:
        candidates=[n for n in archive.namelist() if n.endswith("/"+name) or n==name]
        if len(candidates)!=1:raise FileNotFoundError(
            f"Expected exactly one {name} in {src}; found {len(candidates)}")
        return archive.read(candidates[0])


def list_hessian_npzs(path):
    src=Path(path)
    if src.is_dir():
        return sorted(p.name for p in src.glob("hessians__*.npz"))
    with zipfile.ZipFile(src) as archive:
        return sorted({Path(s).name for s in archive.namelist()
                       if Path(s).name.startswith("hessians__") and s.endswith(".npz")})


def xstd_from_step04(source):
    raw=read_named(source,"all_Y_and_mapping.npz")
    with np.load(io.BytesIO(raw),allow_pickle=False) as f:
        X=np.asarray(f["X_fixed"],dtype=float)
    if X.ndim!=2 or X.shape[1]!=6:
        raise ValueError("Unexpected X_fixed shape in step-04 input")
    std=X.std(axis=0)
    const=std<=1e-12
    return np.where(const,1.,std),const


def score_existing(source,output=None,step04_source=None,
                   min_correlation=.9,endpoint_degrees=10,
                   min_variation=.02,plots=True):
    src=Path(source)
    if not src.exists():raise FileNotFoundError(src)
    if output is None:
        output=src/"rich_sparsity_rescored" if src.is_dir() else (
            src.parent/(src.stem+"__rich_sparsity_rescored"))
    output=Path(output)
    summary=json.loads(read_named(src,"summary.json"))
    xstd=None;std_flags=None;source04=None
    probes=[Path(step04_source)] if step04_source is not None else []
    if step04_source is None:
        if summary.get("source"):probes.append(Path(summary["source"]))
        condition=Path(summary.get("source","")).parent.name
        if condition and condition!=".":
            probes.append(ROOT/"outputs/math/cr3_zx_hessian_threeway"/condition/"repeat01")
    for path in probes:
        if path.exists():
            xstd,std_flags=xstd_from_step04(path)
            source04=str(path)
            break
    if xstd is None:
        print("Step04 X unavailable: standardized-X metrics skipped. "
              "Pass --step04-source to add them.",flush=True)
    output.mkdir(parents=True,exist_ok=True)
    metric_rows=[];attribution_rows=[];produced=[]
    names=list_hessian_npzs(src)
    if not names:raise ValueError("No hessians__*.npz in old experiment result")
    for name in names:
        ident=name[len("hessians__"):-len(".npz")]
        pair=ident.split("_",1)[0]
        if len(pair)!=2 or not pair.isdigit():
            raise ValueError(f"Cannot parse rotating plane in file {name}")
        i,j=int(pair[0]),int(pair[1])
        with np.load(io.BytesIO(read_named(src,name)),allow_pickle=False) as f:
            angles=np.asarray(f["angles"],dtype=float)
            H=np.asarray(f["H"],dtype=float)
            methods=tuple(f["method_names"].tolist())
            targets=tuple(f["target_names"].tolist())
            frac={key:np.asarray(f[f"reference_fraction_defined_{key}"],dtype=float)
                  for key in ("logit","probability")}
        if methods!=SOURCENAMES or targets!=TARGETNAMES:
            raise ValueError(f"Unknown saved source order in {name}: {methods} / {targets}")
        if H.shape!=(len(angles),9,10,10):
            raise ValueError(f"Expected (angles,9,10,10), got {H.shape}")
        if not np.isclose(angles[0],0):
            raise ValueError("Rescore requires 0 degrees as baseline")
        current=[]
        baseline={(s,t):H[0,si*3+ti] for si,s in enumerate(methods)
                  for ti,t in enumerate(targets)}
        for ai,angle in enumerate(angles):
            R=pair_rotation(i,j,float(angle))
            nine={(s,t):H[ai,si*3+ti] for si,s in enumerate(methods)
                  for ti,t in enumerate(targets)}
            refs={t:nine[("analytic",t)] for t in targets}
            validity={t:frac[ORACLE_TARGET[t]] for t in targets}
            for s in methods:
                for t in targets:
                    matrix=nine[(s,t)]
                    metrics=metrics_for_source(matrix,refs[t],validity[t],
                                               ALL_BLOCKS,x_std=xstd)
                    drift=equivariance_error(matrix,baseline[(s,t)],R)
                    for block,values in metrics.items():
                        norm0=float(np.linalg.norm(block_array(baseline[(s,t)],block)))
                        row=dict(plane=ident,parameter_i=i,parameter_j=j,
                                 angle_degrees=float(angle),source=s,target=t,
                                 block=block,**values,equivariance_error=drift,
                                 frobenius_norm_drift=float(values["frob_norm"]/max(norm0,1e-15)-1.))
                        current.append(row)
            for gap in three_way_all(nine,validity,ALL_BLOCKS):
                attribution_rows.append(dict(plane=ident,
                                             angle_degrees=float(angle),**gap))
        metric_rows.extend(current)
        csv_write(output/f"curves__{ident}.csv",current)
        if plots:
            for key in DEFAULT_XZ_SPARSITY_METRICS:
                if key not in next(r for r in current if r["block"]=="ZX"):
                    continue
                file=output/f"xz_sparsity__{ident}__{key}.png"
                plot_xz_sparsity(current,ident,angles,key,file)
                produced.append(file)
        print(f"Rescored {ident}: {len(angles)} angles × 9 methods",flush=True)
    csv_write(output/"all_curves.csv",metric_rows)
    csv_write(output/"three_way_hessian_gaps.csv",attribution_rows)
    best=calculate_best_angles(metric_rows)
    csv_write(output/"best_angles.csv",best)
    checks,groups,ranking=select_sparsity_metrics(
        metric_rows,min_correlation=min_correlation,
        endpoint_tolerance=endpoint_degrees,min_relative_range=min_variation)
    csv_write(output/"sparsity_screen_all.csv",checks)
    csv_write(output/"sparsity_screen_groups.csv",groups)
    csv_write(output/"sparsity_screen_ranking.csv",ranking)
    csv_write(output/"metric_catalog.csv",metric_catalog())
    if produced:plot_index(output,produced)
    info=dict(source=str(src),source_step04=source04,
              X_column_std=xstd,X_std_constant_flags=std_flags,
              original_summary=summary,
              n_hessian_matrices=sum(
                 len(np.load(io.BytesIO(read_named(src,name)))["angles"])*9 for name in names),
              n_metric_rows=len(metric_rows),
              n_three_way_gap_rows=len(attribution_rows),
              n_sparsity_screen_rows=len(checks),
              n_qualifying_screen_rows=sum(bool(r["qualifies"]) for r in checks),
              endpoints="Minima at 0° or 90° within tolerance; maxima do not qualify",
              methodology="From saved H arrays, not recomputed network Y or regressed Hessian",
              failure_warning="Formula-Y regression vs true analytic H is a separate error source from NN-Y fitted vs formula-Y fitted")
    (output/"rescore_summary.json").write_text(json.dumps(
        as_native(info),indent=2,allow_nan=False),encoding="utf-8")
    print("Saved rescore and three-way gap analysis to",output,flush=True)
    return info


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--step04-source",type=Path,
                   help="Original step04 results for standardized-X losses")
    p.add_argument("--min-correlation",type=float,default=.9)
    p.add_argument("--endpoint-degrees",type=float,default=10.)
    p.add_argument("--min-variation",type=float,default=.02)
    p.add_argument("--no-plots",action="store_true")
    args=p.parse_args(argv)
    return score_existing(args.input,args.output,args.step04_source,
                          min_correlation=args.min_correlation,
                          endpoint_degrees=args.endpoint_degrees,
                          min_variation=args.min_variation,
                          plots=not args.no_plots)


if __name__=="__main__":main()
