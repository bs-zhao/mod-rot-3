"""PAIRWISE rotation landscapes around the FIT optimum (diagnostic only).

0 and 90 degrees are exactly equivalent for a full elementwise RMS-L1
sparsity loss up to row signed-permutation: this is a useful consistency
check of the objective implementation, not evidence of true-parameter axes.
"""
from __future__ import annotations
import csv
import numpy as np


def plane_rotate(R,i,j,angle_degrees):
    c=np.cos(np.deg2rad(angle_degrees));s=np.sin(np.deg2rad(angle_degrees))
    G=np.eye(len(R))
    G[i,i]=c;G[j,j]=c
    G[i,j]=s;G[j,i]=-s
    return G@R


def save_planes(out,R,fit,valid):
    from pathlib import Path
    out=Path(out)
    angles=np.linspace(0,90,31)
    records=[]
    for i in range(len(R)):
        for j in range(i+1,len(R)):
            start=len(records)
            for a in angles:
                C=plane_rotate(R,i,j,a)
                records.append(dict(axis_i=i+1,axis_j=j+1,
                                    degrees=float(a),
                                    fit_loss=fit.value(C),
                                    validation_loss=valid.value(C)))
            sub=records[start:]
            if abs(sub[0]["fit_loss"]-sub[-1]["fit_loss"])>1e-7:
                raise RuntimeError("Sparsity objective not invariant to signed 90-degree axis permutation")
    with (out/"plane_landscapes.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for i in range(len(R)):
        for j in range(i+1,len(R)):
            subset=[r for r in records if r["axis_i"]==i+1 and r["axis_j"]==j+1]
            fig,ax=plt.subplots(figsize=(6.8,4.2))
            ax.plot([r["degrees"] for r in subset],
                    [r["fit_loss"] for r in subset],label="FIT")
            ax.plot([r["degrees"] for r in subset],
                    [r["validation_loss"] for r in subset],label="VALID")
            ax.set(xlabel=f"Further rotation of U{i+1} / U{j+1} (degrees)",
                   ylabel="Role-free Hessian sparsity loss",
                   title=f"SO({len(R)}) pairwise curvature landscape")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out/f"plane_u{i+1}_u{j+1}.png",dpi=160)
            plt.close(fig)
    return dict(number_of_planes=len(R)*(len(R)-1)//2,
                angles_per_plane=len(angles),path="plane_landscapes.csv",
                note="0 and 90-degree equal-loss endpoints follow signed-permutation symmetry")
