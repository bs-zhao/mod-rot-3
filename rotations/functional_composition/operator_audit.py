"""FIT-only stability audit for calibrated operator 08.

Report WHICH polynomial modules survive multiple plausible uncertainty scales,
rather than assuming the threshold from one half-split is definitive.
This never influences the fitted R and does not access true CPT parameters.
"""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import numpy as np
from .metrics import calibrated_operator_complexity, _coeff_group_list
from .surrogate import rotate_coefficients


def _module_half_support(proxy,R,config):
    a,b=(rotate_coefficients(coef,R) for coef in proxy.split_coefficients)
    full=rotate_coefficients(proxy.coef,R)
    floor=float(config.get("operator_relative_floor",.025))
    z=float(config.get("operator_noise_z",2.))
    result=[]
    degrees={d:max(float(np.linalg.norm(full[d])),1e-12) for d in full if d}
    for inds,degree,_,norm,_ in _coeff_group_list(full):
        mult=math.factorial(degree)
        for i in set(inds):mult//=math.factorial(inds.count(i))
        mul=math.sqrt(mult)
        a_norm=float(np.linalg.norm(a[degree][inds])*mul)
        b_norm=float(np.linalg.norm(b[degree][inds])*mul)
        uncertainty=max(float(np.linalg.norm(a[degree][inds]-b[degree][inds])*mul/2.),
                        floor*degrees[degree])
        result.append(dict(term="*".join("u"+str(i+1) for i in inds),
                           degree=degree,
                           signal=float(norm),half_a_norm=a_norm,
                           half_b_norm=b_norm,noise=uncertainty,
                           half_a_supported=bool(a_norm>=z*uncertainty),
                           half_b_supported=bool(b_norm>=z*uncertainty),
                           full_supported=bool(norm>=z*uncertainty)))
    return result


def audit(proxy,rotations,config):
    floors=[float(v) for v in config.get("operator_audit_floors",[.01,.025,.05])]
    levels=[float(v) for v in config.get("operator_audit_noise_z",[1.5,2.,2.5,3.])]
    if any(v<=0 for v in floors) or any(v<0 for v in levels):
        raise ValueError("Audit thresholds must be nonnegative, floors positive")
    rows=[];modules=[]
    for name,R in rotations.items():
        for floor in floors:
            for level in levels:
                local=dict(config,operator_relative_floor=floor,operator_noise_z=level)
                loss,summary=calibrated_operator_complexity(proxy,R,local,details=True)
                active=_module_half_support(proxy,R,local)
                agreement=np.mean([m["half_a_supported"]==m["half_b_supported"]
                                   for m in active])
                intersection=sum(m["half_a_supported"] and m["half_b_supported"] for m in active)
                union=sum(m["half_a_supported"] or m["half_b_supported"] for m in active)
                rows.append(dict(rotation=name,relative_floor=floor,noise_z=level,
                                 operator_code=float(loss),
                                 hard_modules=int(summary["hard_modules"]),
                                 total_modules=int(summary["total_modules"]),
                                 half_support_agreement=float(agreement),
                                 half_support_jaccard=float(intersection/max(union,1))))
                if floor==floors[0] and level==levels[0]:
                    modules.extend([dict(rotation=name,**m) for m in active])
    return rows,modules


def save_operator_audit(folder,proxy,rotations,config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not getattr(proxy,"split_coefficients",None):
        return
    folder=Path(folder)
    rows,modules=audit(proxy,rotations,config)
    for filename,data in [
        ("functional__operator_stability.csv",rows),
        ("functional__operator_modules.csv",modules)
    ]:
        if not data:continue
        with (folder/filename).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]))
            writer.writeheader();writer.writerows(data)
    data={"candidate_count":len(rotations),
          "candidate_names":list(rotations),
          "threshold_grid":config.get("operator_audit_noise_z",[1.5,2.,2.5,3.]),
          "relative_floor_grid":config.get("operator_audit_floors",[.01,.025,.05]),
          "caution":"Heuristic FIT-half coefficient noise; threshold robustness does not establish psychological parameter identifiability"}
    (folder/"functional__operator_audit.json").write_text(
        json.dumps(data,indent=2),encoding="utf-8")
    if config.get("make_plots",True) and rows:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for name in rotations:
            subset=[r for r in rows if r["rotation"]==name and
                    r["relative_floor"]==min(config.get("operator_audit_floors",[.01,.025,.05]))]
            ax.plot([r["noise_z"] for r in subset],
                    [r["hard_modules"] for r in subset],marker=".",label=name)
        ax.set_xlabel("Heuristic SNR threshold")
        ax.set_ylabel("Active operator modules")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(folder/"functional__operator_stability.png",dpi=160)
        plt.close(fig)
