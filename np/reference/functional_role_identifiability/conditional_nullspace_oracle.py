"""CR2 oracle conditional nullspace identifiability: NO Neural Process, NO ICA.

Three PSD response-derived information operators are constructed directly
from the exact four-parameter CR2 simulator and its analytical gradients:
  shape           : shape-normalized response function over ALL X (tau null)
  gain_only       : conditional Bernoulli Fisher where both losses = 0 (lambda null)
  gamma_neutral   : conditional Bernoulli Fisher where BOTH p = 0.5 (gamma null)

The third condition uses p=.5 strictly, not clipped 0/1 endpoints.
An objective assigns DIFFERENT orthogonal axes to the three kernels.
When each matrix has exactly the advertised one-dimensional kernel,
the oracle coordinate frame is globally identifiable up to signs and
permutation. Numerical evidence is audited, not assumed.

Observed original 500-X battery and a clearly segregated diagnostic
augmentation are audited separately. Additional stimuli NEVER modify S1.
All outputs go to ignored outputs/choice/CR2/oracle_functional_roles/.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize, linear_sum_assignment
from scipy.special import expit, logit

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(RESEARCH))
from s0_cr2_design_recovery import logits, sample_prior

PARAMS = ("alpha", "lambda_loss", "gamma", "tau")
BLOCKS = ("shape", "gain_only", "gamma_neutral")
NULL_PARAMETER = {"shape": 3, "gain_only": 1, "gamma_neutral": 2}
ASSIGNMENTS = tuple(itertools.permutations(range(4), 3))
PAIRS = tuple(itertools.combinations(range(4), 2))


def _json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def response_and_gradient(X, theta, sd):
    """Analytical logit and d logit/d standardized theta.

    theta shape (subjects,4), X shape (trials,6);
    standardized theta_j=(theta_j-center_j)/sd_j.
    Uses exactly the s0 simulator's 1e-12 probability clipping.
    """
    X = np.asarray(X, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    sd = np.asarray(sd, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 6 or theta.ndim != 2 or theta.shape[1] != 4:
        raise ValueError("Expected X=(trials,6), theta=(subjects,4)")
    if not np.isfinite(X).all() or not np.isfinite(theta).all() or np.any(sd <= 0):
        raise ValueError("Invalid simulator inputs or parameter scales")

    a = theta[:, 0, None]
    lam = theta[:, 1, None]
    gam = theta[:, 2, None]
    tau = theta[:, 3, None]

    def terms(part):
        g = part[:, 0][None, :] / 100.0
        l = part[:, 1][None, :] / 100.0
        p = part[:, 2][None, :]
        p = np.clip(p, 1e-12, 1 - 1e-12)
        z_g = logit(p)
        z_l = logit(np.clip(1-p, 1e-12, 1-1e-12))
        w_g = expit(gam*z_g)
        w_l = expit(gam*z_l)
        gpow = np.power(g, a)
        lpow = np.power(l, a)
        benefit = w_g*gpow
        loss = w_l*lpow
        aval = benefit - lam*loss
        d_alpha = benefit*np.log(np.maximum(g, 1e-300)) - lam*loss*np.log(np.maximum(l, 1e-300))
        d_lam = -loss
        d_gamma = (w_g*(1-w_g)*z_g)*gpow - lam*(w_l*(1-w_l)*z_l)*lpow
        return aval, d_alpha, d_lam, d_gamma

    Aa, dalpha_a, dlambda_a, dgamma_a = terms(X[:, :3])
    Bb, dalpha_b, dlambda_b, dgamma_b = terms(X[:, 3:])
    difference = Aa-Bb
    f = tau*difference
    raw = np.stack((
        tau*(dalpha_a-dalpha_b),
        tau*(dlambda_a-dlambda_b),
        tau*(dgamma_a-dgamma_b),
        difference,
    ), axis=-1)
    return f, raw*sd[None, None, :]


def condition_masks(X):
    """Predeclared physical stimulus conditions, never selected by theta."""
    X = np.asarray(X, float)
    atol = 1e-10
    gain = np.isclose(X[:, 1], 0., atol=atol, rtol=0) & np.isclose(X[:, 4], 0., atol=atol, rtol=0)
    # STRICT: pA=pB=.5 is exactly invariant to gamma even with clipped logits.
    neutral = np.isclose(X[:, 2], .5, atol=atol, rtol=0) & np.isclose(X[:, 5], .5, atol=atol, rtol=0)
    # Additional coverage indicator only: physical endpoint invariance in
    # mathematical CPT differs from 1e-12-clipped implementation.
    endpoints = np.isclose(X[:, 2], 0, atol=atol) | np.isclose(X[:, 2], 1, atol=atol)
    endpoints_b = np.isclose(X[:, 5], 0, atol=atol) | np.isclose(X[:, 5], 1, atol=atol)
    return {"shape": np.ones(len(X), dtype=bool), "gain_only": gain,
            "gamma_neutral": neutral,
            "both_endpoint_stimuli_diagnostic_only": endpoints & endpoints_b}


def _psd(M):
    return (M+M.T)*0.5


def operator_matrices(X, theta, sd, masks=None):
    """Population information: shape derivative Gram + local logistic Fisher.

    For each group, average its trial information within subject, then
    average across subjects. Every subject receives equal total weight.
    """
    if masks is None:
        masks = condition_masks(X)
    f, g = response_and_gradient(X, theta, sd)
    fnorm = np.linalg.norm(f, axis=1)
    if np.any(fnorm <= 1e-14):
        raise ValueError("Zero logit profile; undefined normalized shape")
    s = f/fnorm[:, None]
    proj = np.einsum("nm,nmj->nj", s, g, optimize=True)
    dg = (g-s[:, :, None]*proj[:, None, :])/fnorm[:, None, None]
    operators = {"shape": _psd(np.einsum("nmi,nmj->ij", dg, dg, optimize=True)/(len(theta)*len(X)))}
    fisher_w = expit(f)*(1-expit(f))
    for name in ("gain_only", "gamma_neutral"):
        mask = masks[name]
        if int(mask.sum()) == 0:
            operators[name] = None
            continue
        gg = g[:, mask]
        w = fisher_w[:, mask]
        operators[name] = _psd(np.einsum("nm,nmi,nmj->ij", w, gg, gg, optimize=True)/
                               (len(theta)*int(mask.sum())))
    return operators


def describe_operator(A, expected, trials, min_trials=8, rank_tol=1e-7, min_gap=1e-5):
    result = dict(n_stimuli=int(trials), expected_null_parameter=PARAMS[expected],
                  expected_null_index=int(expected), min_trials=int(min_trials))
    if A is None:
        result.update(available=False, rank3=False, certified_eligible=False,
                      reason="No trials satisfy this condition")
        return result
    eigenvalues, V = np.linalg.eigh(A)
    lammax = max(float(eigenvalues[-1]), 1e-30)
    relative = eigenvalues/lammax
    threshold = rank_tol*lammax
    rank = int(np.count_nonzero(eigenvalues > threshold))
    gap = float(eigenvalues[1]/lammax)
    direction = V[:, 0]
    intended = np.eye(4)[expected]
    align = float(abs(direction@intended))
    residual = float(intended@A@intended/max(float(np.trace(A)), 1e-30))
    operator_ok = (rank == 3 and gap >= min_gap and
                   relative[0] <= rank_tol and align >= 1-1e-5 and residual < 1e-8)
    result.update(available=True, rank3=bool(rank == 3),
                  numeric_rank=rank, relative_eigenvalues=relative.tolist(),
                  eigenvalues=eigenvalues.tolist(), gap_ratio=gap,
                  smallest_eigenvector=direction.tolist(), target_axis_cosine=align,
                  target_rayleigh_normalized=residual,
                  certified_eligible=bool(operator_ok and trials >= min_trials),
                  reason=("eligible" if operator_ok and trials >= min_trials else
                          "Insufficient condition count, eigengap, or one-dimensional nullspace"))
    return result


def audit(X, train_theta, valid_theta, sd, min_trials, gap=1e-5):
    masks = condition_masks(X)
    op_train = operator_matrices(X, train_theta, sd, masks)
    op_valid = operator_matrices(X, valid_theta, sd, masks)
    n = {k: int(masks[k].sum()) for k in masks}
    details = {k: describe_operator(op_train[k], NULL_PARAMETER[k], n[k],
                                    min_trials=min_trials, min_gap=gap) for k in BLOCKS}
    validation = {k: describe_operator(op_valid[k], NULL_PARAMETER[k], n[k],
                                       min_trials=min_trials, min_gap=gap) for k in BLOCKS}
    certified = all(details[k]["certified_eligible"] and validation[k]["certified_eligible"] for k in BLOCKS)
    return op_train, op_valid, dict(coverage=n,train=details,validation=validation,
        empirical_rank_conditions_passed=bool(certified),
        theorem="If all three PSD matrices have distinct, orthogonal 1D kernels equal to physical parameter axes, the 3-assignment nullspace objective has signed-permutation unique zero minima.",
        caveat="Numerical rank checks support but do not prove symbolic rank for arbitrary stimulus populations. Exact gamma-null requires both p=.5 for this clipped implementation.")


def skew(v):
    A = np.zeros((4,4), dtype=float)
    for t,(i,j) in zip(v, PAIRS):
        A[i,j]=t
        A[j,i]=-t
    return A


def rotation(v, base=None):
    Q = expm(skew(v))
    return Q if base is None else Q@base


def assignment_cost(R, operators, eligible, return_assignment=False):
    """Nonnegative objective: distinct rows assigned to distinct nullspaces."""
    active = [k for k in BLOCKS if k in eligible]
    if not active:
        raise ValueError("No eligible operator groups")
    matrices = [operators[name] for name in active]
    normalized = [M/max(float(np.trace(M)), 1e-30) for M in matrices]
    C = np.stack([np.einsum("ai,ij,aj->a", R, M, R) for M in normalized])
    assignments = list(itertools.permutations(range(4), len(active)))
    values = [sum(C[i,a] for i,a in enumerate(assignment)) for assignment in assignments]
    j = int(np.argmin(values))
    cost = float(values[j])
    return (cost, dict(zip(active, assignments[j]))) if return_assignment else cost


def coordinate_stationarity(operators, eligible, h=7e-4):
    """Numerical SO(4) gradient/Hessian at the *true physical* axes."""
    eye = np.eye(6)
    fun = lambda v: assignment_cost(rotation(v), operators, eligible)
    origin = np.zeros(6)
    grad = np.array([(fun(h*e)-fun(-h*e))/(2*h) for e in eye])
    Hess = np.empty((6,6))
    f0 = fun(origin)
    for i in range(6):
        for j in range(i,6):
            if i==j:
                value = (fun(h*eye[i])-2*f0+fun(-h*eye[i]))/h**2
            else:
                value = (fun(h*(eye[i]+eye[j]))-fun(h*(eye[i]-eye[j]))-
                         fun(h*(-eye[i]+eye[j]))+fun(-h*(eye[i]+eye[j])))/(4*h*h)
            Hess[i,j]=Hess[j,i]=value
    return dict(gradient=grad.tolist(),gradient_norm=float(np.linalg.norm(grad)),
                hessian_eigenvalues=np.linalg.eigvalsh(Hess).tolist(),
                hessian_negative_count=int(np.count_nonzero(np.linalg.eigvalsh(Hess)<-1e-5)),
                note="Finite-difference tangent Hessian; stable assignment required locally.")


def _axis_score(R):
    ids,jds=linear_sum_assignment(-np.abs(R))
    return float(np.mean(np.abs(R[ids,jds])))


def optimize(operators, val_operators, eligible, starts, maxiter, seed):
    rng=np.random.default_rng(seed)
    f=lambda R: assignment_cost(R, operators, eligible)
    attempts=[]
    for i in range(starts):
        base=np.eye(4) if i==0 else rotation(rng.normal(0,.7,6))
        obj=lambda v: f(rotation(v,base))
        res=minimize(obj,np.zeros(6),method="L-BFGS-B",
                     options={"maxiter":int(maxiter),"ftol":1e-13,"gtol":1e-10})
        R=rotation(res.x,base)
        loss,assigned=assignment_cost(R,operators,eligible,True)
        attempts.append(dict(start=i,loss=loss,validation_loss=assignment_cost(R,val_operators,eligible),
            success=bool(res.success),iterations=int(res.nit),
            axis_alignment_to_oracle=_axis_score(R),assignment=assigned,rotation=R.tolist()))
    selected=min(attempts,key=lambda obj:obj["loss"])
    return dict(identity_train_loss=f(np.eye(4)),
        identity_validation_loss=assignment_cost(np.eye(4),val_operators,eligible),
        best=selected,starts=attempts,
        global_best_below_identity=bool(selected["loss"] < f(np.eye(4))-1e-8),
        caveat="Multistart is an empirical falsification search; never a numerical proof of global uniqueness.")


def alpha_gamma_rotation(angle):
    R=np.eye(4)
    c,s=np.cos(angle),np.sin(angle)
    R[0,0]=c;R[0,2]=s
    R[2,0]=-s;R[2,2]=c
    return R


def supplement_stimuli(count, seed):
    """Explicit mathematical FEASIBILITY-only additions; never modify observed X."""
    if count<8:raise ValueError("At least 8 per diagnostic stimulus group")
    rng=np.random.default_rng(seed)
    amounts=np.array([5.,10.,15.,20.,30.,40.,60.,80.,100.,125.,150.,200.])
    probs=np.array([.15,.25,.35,.5,.65,.75,.85,.95])
    rows=[]
    for _ in range(count):
        gA,gB=rng.choice(amounts,2,replace=False)
        pA,pB=rng.choice(probs,2,replace=False)
        rows.append([gA,0.,pA,gB,0.,pB])
    for _ in range(count):
        gA,gB,lA,lB=rng.choice(amounts,4,replace=False)
        rows.append([gA,lA,.5,gB,lB,.5])
    return np.asarray(rows,float)


def _digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as file:
        for x in iter(lambda:file.read(1024*1024),b""):
            h.update(x)
    return h.hexdigest()


def check_gradients():
    rng=np.random.default_rng(99)
    theta=sample_prior(5,4,rng)
    x=supplement_stimuli(12,2012)[:8]
    sd=theta.std(0)
    f,g=response_and_gradient(x,theta,sd)
    expected=logits(x[None,:,:],theta[:,None,:])
    if not np.allclose(f,expected,atol=1e-10,rtol=1e-10):
        raise AssertionError("Analytic logit differs from source simulator")
    for j in range(4):
        eps=2e-5*sd[j]
        plus=theta.copy();minus=theta.copy()
        plus[:,j]+=eps;minus[:,j]-=eps
        finite=(logits(x[None,:,:],plus[:,None,:])-logits(x[None,:,:],minus[:,None,:]))/(2e-5)
        if not np.allclose(g[:,:,j],finite,rtol=1e-4,atol=1e-6):
            raise AssertionError(f"Analytic gradient mismatch for {PARAMS[j]}")
    train=sample_prior(40,4,rng)
    val=sample_prior(32,4,rng)
    X=supplement_stimuli(70,2013)
    mt,mv,details=audit(X,train,val,sd,min_trials=8)
    if not details["empirical_rank_conditions_passed"]:
        raise AssertionError("Synthetic identifiable stimulus bank failed rank check: "+json.dumps(details["train"]))
    I=np.eye(4)
    if assignment_cost(I,mt,set(BLOCKS)) > 1e-9:
        raise AssertionError("Physical parameter identity not zero")
    # The old two-nullspace combination cannot distinguish alpha/gamma.
    if assignment_cost(alpha_gamma_rotation(.5),mt,{"shape","gain_only"})>1e-9:
        raise AssertionError("Two nullspaces should NOT identify alpha/gamma")
    if assignment_cost(alpha_gamma_rotation(.5),mt,set(BLOCKS))<1e-6:
        raise AssertionError("Third nullspace did not distinguish alpha/gamma")
    print("Analytic gradients + one-dimensional kernel rank + alpha/gamma counterexample PASS")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=("all","audit","optimize","selftest"),default="all")
    p.add_argument("--dataset",default=None,help="CR2 S1 generated dataset NPZ; read-only")
    p.add_argument("--subjects",type=int,default=96,help="Oracle subjects, divided equally for train/validation")
    p.add_argument("--supplement-count",type=int,default=64,help="EXTRA group size for diagnostic augmented bank")
    p.add_argument("--min-condition-trials",type=int,default=8)
    p.add_argument("--minimum-rank-gap",type=float,default=1e-5)
    p.add_argument("--starts",type=int,default=10)
    p.add_argument("--maxiter",type=int,default=200)
    p.add_argument("--seed",type=int,default=20261023)
    a=p.parse_args()
    if a.stage=="selftest":
        check_gradients()
        return
    if (min(a.subjects,a.supplement_count,a.min_condition_trials,a.starts,a.maxiter)<1 or
        a.subjects<16 or a.minimum_rank_gap<=0):
        p.error("Require subjects>=16 and positive budgets/rank gap")
    data=Path(a.dataset) if a.dataset else ROOT/"outputs"/"choice"/"CR2"/"s1_s4"/"4P"/"s1_generate"/"s1_generate__dataset.npz"
    if not data.is_file():raise FileNotFoundError(f"CR2 dataset not found: {data}")
    with np.load(data,allow_pickle=False) as f:
        X=np.asarray(f["x"][:500],float)
        theta=np.asarray(f["parameters"],float)
    if X.shape != (500,6) or theta.ndim!=2 or theta.shape[1]!=4:
        raise ValueError("Input must contain CR2-4P fixed 500 x 6 bank and 4P subjects")
    sd=theta.std(axis=0,ddof=0)
    rng=np.random.default_rng(a.seed)
    ids=rng.choice(len(theta),size=min(a.subjects,len(theta)),replace=False)
    mid=len(ids)//2
    if mid<8 or len(ids)-mid<8:raise ValueError("Insufficient oracle train/validation subjects")
    train=theta[ids[:mid]]
    valid=theta[ids[mid:]]
    out=ROOT/"outputs"/"choice"/"CR2"/"oracle_functional_roles"/"conditional_nullspace"
    out.mkdir(parents=True,exist_ok=True)
    report=dict(source="CR2 4P exact simulator",dataset=str(data),
                dataset_sha256=_digest(data),seed=a.seed,
                sample_size=dict(train=len(train),validation=len(valid)),
                parameter_order=list(PARAMS),standardization_sd=sd.tolist(),
                observed_stimulus_count=len(X),
                no_NP=True,no_ICA=True,
                augmented_stimuli_not_used_in_training_data=True,
                conditions_predefined_by_X_and_independent_of_true_theta=True,
                assumptions="Fixed orthogonal rotation of individually standardized true parameters; not a nonlinear NP embedding.")
    banks={"observed_500":X,
           "observed_plus_diagnostic_stimuli":np.vstack((X,supplement_stimuli(a.supplement_count,a.seed+17)))}
    for label,bank in banks.items():
        train_op,val_op,diag=audit(bank,train,valid,sd,a.min_condition_trials,a.minimum_rank_gap)
        eligible=[k for k in BLOCKS if diag["train"][k]["certified_eligible"] and
                  diag["validation"][k]["certified_eligible"]]
        diag["eligible_blocks"]=eligible
        diag["no_unique_4_axis_identification"]=not diag["empirical_rank_conditions_passed"]
        diag["basis_uniqueness_reason"]=(
            "All three independent one-dimensional kernels detected; signed permutation is unique under stated mathematical assumptions"
            if diag["empirical_rank_conditions_passed"] else
            "One or more of the three independent one-dimensional kernels could not be certified")
        report[label]=diag
        if a.stage in ("all","optimize") and eligible:
            tmp=dict(eligible=eligible,
                     tangent_at_oracle=coordinate_stationarity(train_op,set(eligible)),
                     optimized=optimize(train_op,val_op,set(eligible),a.starts,a.maxiter,a.seed+len(label)))
            family=[]
            for angle in np.linspace(0,np.pi/2,7):
                R=alpha_gamma_rotation(angle)
                family.append(dict(angle_degrees=float(np.rad2deg(angle)),
                    objective=assignment_cost(R,train_op,set(eligible))))
            tmp["alpha_gamma_rotation_family"]=family
            report[label]["rotations"]=tmp
        print(f"{label}: coverage={diag['coverage']}; usable={eligible}; "
              f"full_identifiability_rank_gate={diag['empirical_rank_conditions_passed']}",flush=True)
    path=out/"conditional_nullspace__report.json"
    path.write_text(json.dumps(report,indent=2,default=_json),encoding="utf-8")
    print(f"Saved independent CR2 oracle nullspace audit: {path}",flush=True)


if __name__=="__main__":
    main()
