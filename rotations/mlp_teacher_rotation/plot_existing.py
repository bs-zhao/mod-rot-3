"""Add U-U and theta-U plots to existing finished mlp_teacher_rotation runs.

No decoder, fitting, optimizers, or theta-informed rotation selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"np"/"src"))

import numpy as np
from scipy.optimize import linear_sum_assignment
from rotations.mlp_teacher_rotation.experiment import corr
from rotations.mlp_teacher_rotation.u_correlations import (
    save_u_correlation_audit,save_theta_u_pairwise_scatter,
)


def add_plots_to_existing_run(folder: Path) -> list[Path]:
    folder=Path(folder)
    meta_path=folder/"rotation__fit.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Rotation fit metadata missing: {meta_path}")
    meta=json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("package")!="mlp_teacher_rotation":
        raise ValueError(f"Not an mlp_teacher_rotation output: {folder}")
    test_path=folder/"rotation__test_truth.npz"
    theta_test=test_ids=None
    if test_path.is_file():
        with np.load(test_path,allow_pickle=False) as f:
            theta_test=np.asarray(f["theta_test"],float)
            test_ids=np.asarray(f["test_ids"],int)
    names=None
    evaluated=folder/"rotation__test_evaluation.json"
    if evaluated.is_file():
        names=json.loads(evaluated.read_text(encoding="utf-8")).get("parameter_names")
    if theta_test is not None and (names is None or len(names)!=theta_test.shape[1]):
        if meta.get("model")=="CR3":
            names=["alpha","lambda_loss","gamma","tau"]
        else:
            from modlearn_rot.generators.registry import get_model
            names=list(get_model(meta["model"]).parameter_names)

    found=[]
    for target in [folder,*sorted(folder.glob("candidate__*"))]:
        u_path=target/"rotation__U.npz"
        if not u_path.is_file():
            continue
        with np.load(u_path,allow_pickle=False) as f:
            U=np.asarray(f["U"],float)
            train=np.asarray(f["train_ids"],int)
            validation=np.asarray(f["validation_ids"],int)
            test=np.asarray(f["test_ids"],int)
        save_u_correlation_audit(target,U,train,validation,test,make_plots=True)
        if theta_test is not None:
            if not np.array_equal(test,test_ids):
                raise RuntimeError(f"TEST subject ids do not match: {target}")
            c=corr(theta_test,U[test])
            ri,ci=linear_sum_assignment(-(c**2))
            matches=dict(zip(map(int,ri),map(int,ci)))
            save_theta_u_pairwise_scatter(
                target,theta_test,U[test],names,corr_matrix=c,matching=matches)
        found.append(target)
        print(f"[PLOTS] {target} | TEST U-U scatter + U-U heatmap"
              +(" + TEST theta-U scatter" if theta_test is not None
                 else " (no TEST truth: theta-U unavailable)"))
    if not found:
        raise FileNotFoundError(f"No saved rotation__U.npz under {folder}")
    return found


def main(argv=None):
    parser=argparse.ArgumentParser(
        description="Post-hoc U correlation/scatter plots for existing completed runs")
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-dir",type=Path,help="One existing run_<hash> directory")
    group.add_argument("--model",choices=("C1","C2","C3","CR3"))
    parser.add_argument("--condition",help="Required with --model; scans existing run_*")
    parser.add_argument("--repeat",type=int,default=1)
    args=parser.parse_args(argv)
    if args.run_dir is not None:
        targets=[args.run_dir]
    else:
        if not args.condition:
            parser.error("--condition is required with --model")
        base=(ROOT/"outputs"/"choice"/args.model/"rotations"/
              "mlp_teacher_rotation"/args.condition/f"repeat{args.repeat:02d}")
        targets=sorted(p for p in base.glob("run_*") if p.is_dir())
        if not targets:
            raise FileNotFoundError(f"No run_* directories found under {base}")
    for path in targets:
        add_plots_to_existing_run(path)
    print(f"[DONE] plotted {len(targets)} existing run(s), no refitting")


if __name__=="__main__":
    main()
