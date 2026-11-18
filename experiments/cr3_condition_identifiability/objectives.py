"""Rotation-only A/B/C losses from exact analytic CR3 derivative moments.

No target theta labels, prescribed true axis, oracle rotations, or NP z
appear inside any candidate objective. True axes are evaluated ONLY after
loss minimization. Fisher information is rotation-invariant in spectrum,
so only its condition-dependent ENTRY STRUCTURE can select an axis.
"""
from __future__ import annotations

from itertools import combinations
import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment, minimize

from .design import NAMES

C_KINDS = ("ambient", "tangent", "finite_tangent", "finite_boundary", "finite_cross")
METRICS = (
    "A_jointdiag", "A_two_condition_null", "A_jointdiag_plus_null",
    "B_H_offdiag", "B_H_top4_tail", "B_G_top2_tail",
    "C_ambient_top1", "C_tangent_top1",
    "C_finite_tangent_top1", "C_finite_boundary_top1",
    "C_finite_cross_top1",
)


def moments(design, fields):
    """Per-condition sufficient moments preserving missing-X-domain masks.

    K[category,condition,X,a,b] is *conditional* on validity; support
    [category,condition,X] records the fraction valid. No boundary value
    is ever entered as a fabricated zero derivative.
    """
    S, K = fields["G"].shape[:2]
    fis = np.zeros((4,4,4))
    gram = np.zeros_like(fis)
    h4 = np.zeros((4,4,4,4,4))
    covs = {name: np.zeros((4,6,4,4)) for name in C_KINDS}
    supports = {name: np.zeros((4,6)) for name in C_KINDS}
    counts = np.bincount(design.codes,minlength=4)
    for s in range(4):
        rows = design.codes == s
        ntr = int(rows.sum())
        if not ntr:
            continue
        gg = fields["G"][:,rows,:].reshape(-1,4)
        gg_eta = fields["G_logit"][:,rows,:].reshape(-1,4)
        hh = fields["H"][:,rows,:,:].reshape(-1,4,4)
        ww = fields["probability_weight"][:,rows].reshape(-1)
        gram[s] = gg.T @ gg / len(gg)
        fis[s] = np.einsum("na,nb,n->ab",gg_eta,gg_eta,ww,optimize=True) / len(gg_eta)
        h4[s] = np.einsum("nab,ncd->abcd",hh,hh,optimize=True) / len(hh)
        for j in range(6):
            masks = {
                "ambient": fields["ambient_valid"][:,j],
                "tangent": fields["tangent_valid"][:,j],
                "finite_tangent": fields["finite_kind"][:,j] == 0,
                "finite_boundary": fields["finite_kind"][:,j] == 1,
                "finite_cross": fields["finite_kind"][:,j] == 2,
            }
            for kind, mask in masks.items():
                eligible = rows & mask
                n = int(eligible.sum())
                supports[kind][s,j] = n / ntr
                if n == 0:
                    continue
                tensor = fields["UX"] if kind in ("ambient","tangent") else fields["finite_UX"]
                vec = tensor[:,eligible,:,j].reshape(-1,4)
                covs[kind][s,j] = vec.T @ vec / len(vec)
    return dict(F=fis, G_cov=gram, H4=h4, C_cov=covs,
                C_support=supports, counts=counts,
                channel=fields["channel"], x_scale=fields["x_scale"])


def selection_weights(counts, mixture):
    if mixture == "empirical":
        return counts / counts.sum()
    if mixture == "balanced":
        return np.ones(4) / 4
    if mixture in NAMES:
        w = np.zeros(4); w[NAMES.index(mixture)] = 1.
        return w
    raise ValueError(mixture)


def _rot_matrix(m,R):
    return R @ m @ R.T


def score_all(stats,R,mixture="empirical"):
    """Lower is simpler. All ratios scale-free within their defined families."""
    R = np.asarray(R,dtype=float)
    if R.shape!=(4,4) or not np.allclose(R@R.T,np.eye(4),atol=1e-7):
        raise ValueError("R must be a real 4x4 orthogonal rotation")
    w = selection_weights(stats["counts"],mixture)
    F = stats["F"]; G=stats["G_cov"]; H=stats["H4"]
    eps=1e-15
    joint_num=joint_den=0.
    nulls = []
    h_off=h_total=h_tail=0.
    grad_tail=grad_total=0.
    ctail={k:0. for k in C_KINDS}
    ctotal={k:0. for k in C_KINDS}
    for s in range(4):
        if w[s]==0:continue
        Fr=_rot_matrix(F[s],R)
        joint_num += w[s]*(np.sum(Fr**2)-np.sum(np.diag(Fr)**2))
        joint_den += w[s]*np.sum(F[s]**2)
        # Both conditions used by the structural null-axis objective:
        # gain-only and gamma-neutral. No named parameter axis appears.
        Gr=_rot_matrix(G[s],R)
        e=np.maximum(np.diag(Gr),0)
        grad_tail+=w[s]*(np.sum(e)-np.sum(np.sort(e)[-2:]))
        grad_total+=w[s]*np.sum(e)
        hr = np.einsum("ia,jb,ic,jd,abcd->ij",R,R,R,R,H[s],optimize=True)
        hr = np.maximum(hr,0)
        h_off += w[s]*(np.sum(hr)-np.trace(hr))
        h_total += w[s]*np.sum(hr)
        h_tail+=w[s]*(np.sum(hr)-np.sum(np.sort(hr.ravel())[-4:]))
        for kind in C_KINDS:
            for j in range(6):
                support=stats["C_support"][kind][s,j]
                if support==0:continue
                c=_rot_matrix(stats["C_cov"][kind][s,j],R)
                diag=np.maximum(np.diag(c),0)
                energy=np.sum(diag)
                ctail[kind]+=w[s]*support*(energy-np.max(diag))
                ctotal[kind]+=w[s]*support*energy
    # Cross-condition null directions: choose TWO DISTINCT coordinate axes.
    # Both patterns must coexist; an axis cannot satisfy both at once.
    fn=np.maximum(np.diag(_rot_matrix(F[0],R)),0) / max(np.trace(F[0]),eps)
    gn=np.maximum(np.diag(_rot_matrix(F[1],R)),0) / max(np.trace(F[1]),eps)
    two_null=float(min(fn[i]+gn[j] for i in range(4) for j in range(4) if i!=j))
    joint = float(max(joint_num,0)/max(joint_den,eps))
    out={
        "A_jointdiag":joint,
        "A_two_condition_null":two_null,
        "A_jointdiag_plus_null":joint+two_null,
        "B_H_offdiag":float(max(h_off,0)/max(h_total,eps)),
        "B_H_top4_tail":float(max(h_tail,0)/max(h_total,eps)),
        "B_G_top2_tail":float(max(grad_tail,0)/max(grad_total,eps)),
    }
    for name in C_KINDS:
        out[f"C_{name}_top1"]=float(max(ctail[name],0)/max(ctotal[name],eps)) if ctotal[name]>eps else float("nan")
    return out


def fisher_diagnostics(stats):
    """Sample-averaged Fisher eigenstructure; not an axis-recovery proof."""
    result = {}
    for s,name in enumerate(NAMES):
        F = stats["F"][s]
        ev,vec=np.linalg.eigh(F)
        result[name]=dict(
            trials=int(stats["counts"][s]),F=F.tolist(),
            eigenvalues=ev.tolist(),
            trace=float(np.trace(F)),
            rank_relative_1e_8=int(np.count_nonzero(ev>max(ev[-1],1e-15)*1e-8)),
            smallest_eigenvector=vec[:,0].tolist())
    return result


def skew_from_six(a):
    a=np.asarray(a,dtype=float)
    if a.size!=6:raise ValueError("six plane angles")
    m=np.zeros((4,4))
    for value,(i,j) in zip(a,combinations(range(4),2)):
        m[i,j]=value
        m[j,i]=-value
    return m


def matrix_from_params(a):
    return expm(skew_from_six(a))


def plane_rotation(i,j,deg):
    rad=np.deg2rad(deg)
    R=np.eye(4)
    R[i,i]=R[j,j]=np.cos(rad)
    R[i,j]=-np.sin(rad)
    R[j,i]=np.sin(rad)
    return R


def haar_rotation(rng):
    a=rng.normal(size=(4,4))
    q,r=np.linalg.qr(a)
    q=q@np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q)<0:q[:,0]*=-1
    return q


def candidate_rotations(seed=20261117,random_count=96):
    rng=np.random.default_rng(seed)
    yield "identity",np.eye(4),{"kind":"truth_reference"}
    for i,j in combinations(range(4),2):
        for deg in (15,30,45,60,75,90):
            yield f"plane_{i+1}{j+1}_{deg}",plane_rotation(i,j,deg),{
                "kind":"plane","axes":[i,j],"degrees":deg}
    for i in range(random_count):
        yield f"haar_{i+1:04d}",haar_rotation(rng),{"kind":"haar"}


def alignment(R):
    """Only post-hoc: recover theta axes up to signed permutation."""
    row,col=linear_sum_assignment(-np.abs(R))
    matched=np.abs(R[row,col])
    return dict(axis_mean_abs_cos=float(matched.mean()),
                axis_min_abs_cos=float(matched.min()),
                axis_max_angle_deg=float(np.degrees(np.arccos(np.clip(matched.min(),0,1)))),
                axis_mean_angle_deg=float(np.degrees(np.arccos(np.clip(matched,0,1))).mean()),
                permutation=col.tolist())


def scan(stats,mixtures=("empirical","balanced"),random_count=96,
         seed=20261117,optimize=True,restarts=2,maxiter=90,
         optimized_metrics=("A_jointdiag_plus_null","B_H_offdiag",
                            "B_H_top4_tail","C_tangent_top1",
                            "C_finite_tangent_top1","C_finite_boundary_top1")):
    """Enumerate 6 planes + 4D Haar, refine optima on SO(4).

    Targets cannot access theta labels / true R. The identity rotation
    is included only as an explicitly labeled *post-hoc reference*.
    """
    rng=np.random.default_rng(seed+31)
    rot=list(candidate_rotations(seed,random_count))
    rows=[]
    cache={}
    for label,R,meta in rot:
        for mixture in mixtures:
            values=score_all(stats,R,mixture)
            cache[(label,mixture)]=values
            align=alignment(R)
            rows.extend(dict(rotation=label,kind=meta["kind"],mixture=mixture,
                             metric=k,score=float(v),**align)
                        for k,v in values.items())
    if optimize:
        for mixture in mixtures:
            for metric in optimized_metrics:
                if metric not in METRICS:raise ValueError(f"unrecognized objective {metric}")
                # Optimization is fully oracle-blind: independent random
                # SO(4) parameter seeds. Identity / plane sweeps are
                # diagnostic baselines only, NEVER search initializations.
                starts=rng.uniform(-np.pi,np.pi,size=(restarts,6))
                for rep,initial in enumerate(starts):
                    def objective(params):
                        R=matrix_from_params(params)
                        return score_all(stats,R,mixture)[metric]
                    result=minimize(objective,initial,method="BFGS",
                                    options={"maxiter":maxiter,"gtol":1e-7})
                    R=matrix_from_params(result.x)
                    val=score_all(stats,R,mixture)[metric]
                    label=f"optimized_{metric}_{mixture}_r{rep+1}"
                    rot.append((label,R,{"kind":"optimized","success":bool(result.success),
                                          "iterations":int(result.nit)}))
                    cache[(label,mixture)]=score_all(stats,R,mixture)
                    al=alignment(R)
                    rows.extend(dict(rotation=label,kind="optimized",mixture=mixture,
                                     metric=k,score=float(v),**al)
                                for k,v in cache[(label,mixture)].items())
    best={}
    for mixture in mixtures:
        for metric in METRICS:
            # The identity and oracle-centered pairwise sweeps are
            # assessment probes, not candidates for 'recovered' R.
            relevant = [r for r in rows if r["metric"]==metric and
                        r["mixture"]==mixture and r["kind"] in ("haar","optimized")
                        and np.isfinite(r["score"])]
            if not relevant:continue
            choice=min(relevant,key=lambda r:r["score"])
            identity=cache[("identity",mixture)][metric]
            rr=next(R for label,R,_ in rot if label==choice["rotation"])
            best[f"{mixture}/{metric}"]=dict(
                best=choice,identity_score=float(identity),
                identity_minus_search_loss=float(identity-choice["score"]),
                best_rotation_matrix=rr.tolist(),
                n_evaluated=len(relevant))
    return rows,best
