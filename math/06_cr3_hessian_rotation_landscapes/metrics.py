"""Hessian simplicity, rotation landscape, and oracle-agreement metrics.

All scores are computed from the SAME 10x10 *mean Hessian* for each
source/Y/angle. In a symmetric block, full Frobenius norms include the
usual two symmetric off-diagonal entries, while pattern scoring counts
each unique pair once (as in step 05).
"""
from __future__ import annotations
import numpy as np
from patterns import score
from sparsity import rich_sparsity

BLOCKS=("ZZ","ZX","XX","FULL","ZZ_diag","ZZ_offdiag","XX_diag","XX_offdiag")


def block_array(H,block):
    H=np.asarray(H,dtype=float)
    if H.shape!=(10,10):raise ValueError("Expected 10x10 Hessian")
    if block.startswith("ZZ"):M=H[:4,:4]
    elif block.startswith("XX"):M=H[4:,4:]
    elif block=="ZX":return H[:4,4:]
    elif block=="FULL":M=H
    else:raise KeyError(block)
    if block.endswith("_diag"):return np.diag(M)
    if block.endswith("_offdiag"):
        rows,cols=np.triu_indices(len(M),k=1)
        return M[rows,cols]
    return M


def structure(H,block,*,x_std=None):
    M=block_array(H,block)
    a=np.asarray(M,dtype=float)
    vals=a.ravel()
    norm=np.linalg.norm(vals)
    n=len(vals)
    abs_a=np.abs(vals)
    if norm<1e-15:
        result=dict(frob_norm=0.,absolute_l1=0.,
                    normalized_l1=None,max_abs_over_frob=None,
                    offdiag_energy_ratio=None,row_concentration=None,
                    row_entropy=None,effective_rank=None,
                    signed_mean=float(vals.mean()))
        result.update(rich_sparsity(a,block,x_std=x_std))
        return result
    result=dict(
        frob_norm=float(norm),
        absolute_l1=float(abs_a.sum()),
        normalized_l1=float(abs_a.sum()/(np.sqrt(n)*norm)),
        max_abs_over_frob=float(abs_a.max()/norm),
        signed_mean=float(vals.mean()),
        offdiag_energy_ratio=None,
        row_concentration=None,
        row_entropy=None,
        effective_rank=None,
    )
    if M.ndim!=2:return result
    if block in ("ZZ","XX","FULL"):
        total=float(np.sum(M*M))
        off=M.copy()
        np.fill_diagonal(off,0.)
        result["offdiag_energy_ratio"]=float(np.sqrt(np.sum(off*off)/total))
    # Row concentration is meaningful for parameter rows of H_ZX;
    # it quantifies whether effects occupy one or many latent directions.
    if block=="ZX":
        row=np.linalg.norm(M,axis=1)
        if row.sum()>1e-15:
            prob=row/row.sum()
            q=prob[prob>0]
            result["row_concentration"]=float(np.max(prob))
            result["row_entropy"]=float(-np.sum(q*np.log(q))/np.log(len(row)))
    singular=np.linalg.svd(M,compute_uv=False)
    q=(singular**2)
    if q.sum()>1e-18:
        q=q/q.sum()
        positive=q[q>0]
        result["effective_rank"]=float(np.exp(-np.sum(positive*np.log(positive))))
    result.update(rich_sparsity(a,block,x_std=x_std))
    return result


def compare(H,analytic_H,validity,block):
    metrics=score(analytic_H,H,validity,block)
    return dict(
        cosine_to_analytic=metrics.get("cosine"),
        pearson_to_analytic=metrics.get("pearson"),
        abs_rank_to_analytic=metrics.get("abs_rank_spearman"),
        relative_error_to_analytic=metrics.get("relative_frobenius"),
        mae_to_analytic=metrics.get("mae"),
        sign_to_analytic=metrics.get("important_sign_agreement"),
        largest_error_fraction=metrics.get("top1_squared_error_fraction"),
        n_elements=metrics.get("n"),
        n_important=metrics.get("important_count"),
    )


def metrics_for_source(H,analytic_H,validity,blocks=BLOCKS,*,x_std=None):
    return {
        block:{**structure(H,block,x_std=x_std),**compare(H,analytic_H,validity,block)}
        for block in blocks
    }


def equivariance_error(H_at_angle,H_at_zero,R):
    """Relative error after mapping rotated fitted H back to anchor axes."""
    from rotation import rotated_to_star
    back=rotated_to_star(H_at_angle,R)
    n=np.linalg.norm(H_at_zero)
    return float(np.linalg.norm(back-H_at_zero)/max(n,1e-15))
