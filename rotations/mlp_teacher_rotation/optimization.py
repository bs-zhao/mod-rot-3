"""Independent suite's structural candidate registry and SO(k) optimizers.

All inputs originate from frozen NP FIT ZXY or the frozen decoder. No theta,
oracle model equation, or TEST parameter is accepted by this module.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from . import metrics as pm

PRIMITIVES = (
    "operator08", "fingerprint11", "jointdiag13", "separation07",
    "hypergraph09", "polysparse16",
    "teacher_fp_probability", "teacher_fp_logit",
    "teacher_jd_probability", "teacher_jd_logit",
    "teacher_fp_centered", "teacher_jd_centered",
    "teacher_channel_sparse_probability", "teacher_channel_sparse_logit",
    "teacher_global_jd",
)
COMPOSITES = (
    "adaptive_profile", "adaptive_teacher_probability",
    "adaptive_teacher_logit", "hybrid_profile_teacher",
)
ALL_OBJECTIVES = PRIMITIVES + COMPOSITES


def rotation_from_angles(angles, k):
    angles = np.asarray(angles, float)
    if angles.shape != (k*(k-1)//2,):
        raise ValueError("Wrong number of SO(k) generator parameters")
    A = np.zeros((k, k), dtype=float)
    i, j = np.triu_indices(k, 1)
    A[i, j] = angles
    A[j, i] = -angles
    return expm(A)


class MetricBank:
    def __init__(self, proxy, direct, config):
        self.proxy = proxy
        self.direct = direct
        self.config = config
        self.k = proxy.k
        rng = np.random.default_rng(int(config["seed"])+918)
        subjects = proxy.q_train[proxy.fit_ids]
        self.anchor = subjects[np.sort(rng.choice(
            len(subjects), min(len(subjects), int(config["anchor_subjects"])), replace=False))]
        self.gp = np.asarray(direct["probability"], float)
        self.gl = np.asarray(direct["logit"], float)
        if self.gp.ndim != 3 or self.gp.shape[1] != self.k:
            raise ValueError("Direct teacher gradients have wrong shape")
        if self.gp.shape != self.gl.shape:
            raise ValueError("Probability/logit teacher gradients differ in shape")
        # E[G_a(X)G_b(X)] per stimulus feature, fixed BEFORE candidate rotations.
        self.cov = {}
        for name, grad in (("probability", self.gp), ("logit", self.gl)):
            self.cov[name] = np.einsum("nil,njl->lij", grad, grad, optimize=True)/len(grad)
        centered = self.gp-self.gp.mean(axis=0, keepdims=True)
        self.cov["centered"] = np.einsum(
            "nil,njl->lij", centered, centered, optimize=True)/len(centered)
        self.base = {key: self.primitive(np.eye(self.k), key) for key in PRIMITIVES}
        self.baseline_composite = {key: self.score(np.eye(self.k), key) for key in COMPOSITES[:-1]}
        self.baseline_composite["hybrid_profile_teacher"] = self.score(np.eye(self.k), "hybrid_profile_teacher")

    @staticmethod
    def _fingerprint(grad, R):
        rot = np.einsum("ia,nal->nil", R, grad, optimize=True)
        # K x L x L channel second moments; eigenspectrum may have zero modes.
        c = np.einsum("nil,nim->ilm", rot, rot, optimize=True)/len(rot)
        eig = np.maximum(np.linalg.eigvalsh(c), 0)
        total = eig.sum(axis=1, keepdims=True)
        p = np.divide(eig, total, out=np.zeros_like(eig), where=total>1e-14)
        entropy = -np.sum(np.where(p>1e-12, p*np.log(np.maximum(p, 1e-300)), 0.), axis=1)
        effective = np.where(total[:, 0]>1e-14, np.exp(entropy), 0.)
        return float(np.sum(np.maximum(effective-1., 0.)))

    @staticmethod
    def _jointdiag(cov, R):
        rotated = np.einsum("ia,lab,jb->lij", R, cov, R, optimize=True)
        off = rotated.copy()
        diag = np.arange(len(R))
        off[:,diag,diag] = 0
        denom = np.sum(cov**2, axis=(1,2))
        valid = denom>1e-12
        if not np.any(valid):
            return 0.
        return float(np.mean(np.sum(off[valid]**2, axis=(1,2))/denom[valid]))

    @staticmethod
    def _axis_channel_sparsity(grad, R):
        rot = np.einsum("ia,nal->nil", R, grad, optimize=True)
        # L1/L2 across latent axes for each (subject, X) -- non-invariant.
        norm = np.sqrt(np.sum(rot**2,axis=1))
        active = norm>1e-10
        if not np.any(active):
            return 0.
        ratio = np.sum(np.abs(rot),axis=1)/(norm+1e-12)
        return float(np.mean((ratio-1.)[active]))

    def primitive(self, R, name):
        R = np.asarray(R, float)
        if name == "operator08":
            return float(pm.calibrated_operator_complexity(self.proxy,R,self.config))
        if name == "fingerprint11":
            return float(pm.fingerprint_and_drift(self.proxy,R,self.anchor)[0])
        if name == "jointdiag13":
            return float(pm.contextual_jointdiag(self.proxy,R,self.anchor))
        if name in ("separation07", "hypergraph09", "polysparse16"):
            coeff = pm.rotate_coefficients(self.proxy.coef,R)
            if name == "separation07":
                return float(pm.interaction_rank(coeff)[0])
            support = pm.support_complexity(coeff,float(self.config["support_threshold"]))
            return float(support["hypergraph"] if name == "hypergraph09" else support["sparsity"])
        if name == "teacher_fp_probability":
            return self._fingerprint(self.gp,R)
        if name == "teacher_fp_logit":
            return self._fingerprint(self.gl,R)
        if name == "teacher_fp_centered":
            return self._fingerprint(self.gp-self.gp.mean(0,keepdims=True),R)
        if name in ("teacher_jd_probability", "teacher_jd_logit", "teacher_jd_centered"):
            mode = {"teacher_jd_probability":"probability",
                    "teacher_jd_logit":"logit","teacher_jd_centered":"centered"}[name]
            return self._jointdiag(self.cov[mode],R)
        if name == "teacher_global_jd":
            return self._jointdiag(self.cov["probability"].mean(0,keepdims=True),R)
        if name == "teacher_channel_sparse_probability":
            return self._axis_channel_sparsity(self.gp,R)
        if name == "teacher_channel_sparse_logit":
            return self._axis_channel_sparsity(self.gl,R)
        raise KeyError(f"Unknown primitive {name!r}")

    def _adaptive(self,R,j_name,f_name):
        j0=max(abs(self.base[j_name]), 1e-8)
        f0=max(abs(self.base[f_name]), 1e-8)
        return float(self.primitive(R,j_name)+(f0/j0)*self.primitive(R,f_name))

    def score(self,R,name):
        if name in PRIMITIVES:
            return self.primitive(R,name)
        if name=="adaptive_profile":
            return self._adaptive(R,"jointdiag13","fingerprint11")
        if name=="adaptive_teacher_probability":
            return self._adaptive(R,"teacher_jd_probability","teacher_fp_probability")
        if name=="adaptive_teacher_logit":
            return self._adaptive(R,"teacher_jd_logit","teacher_fp_logit")
        if name=="hybrid_profile_teacher":
            a=self._adaptive(R,"jointdiag13","fingerprint11")
            b=self._adaptive(R,"teacher_jd_probability","teacher_fp_probability")
            # Each score has fixed identity normalization; no theta or TEST use.
            return .5*a/max(self.baseline_composite["adaptive_profile"],1e-8)+.5*b/max(
                self.baseline_composite["adaptive_teacher_probability"],1e-8)
        raise KeyError(f"Unknown objective: {name}")

    def diagnose(self,R):
        values={name:self.primitive(R,name) for name in PRIMITIVES}
        values.update({name:self.score(R,name) for name in COMPOSITES})
        return values


def optimize_one(bank,name,config):
    if name not in ALL_OBJECTIVES:
        raise ValueError(f"Unknown metric: {name}")
    k=bank.k
    starts=int(config["starts"])
    maxiter=int(config["maxiter"])
    if starts<1 or maxiter<1:
        raise ValueError("starts and maxiter must be positive")
    rng=np.random.default_rng(int(config["seed"])+sum(map(ord,name)))
    options={"maxiter":maxiter}
    runs=[]
    evaluations=0

    def objective(angle):
        nonlocal evaluations
        evaluations+=1
        return bank.score(rotation_from_angles(angle,k),name)

    for i in range(starts):
        guess=np.zeros(k*(k-1)//2) if i==0 else rng.normal(0,.4,k*(k-1)//2)
        fit=minimize(objective,guess,method=str(config["optimizer"]),options=options)
        if np.isfinite(fit.fun):
            runs.append((float(fit.fun),rotation_from_angles(fit.x,k),fit))
    if not runs:
        raise RuntimeError(f"No finite optimizer solution for {name}")
    value,R,fit=min(runs,key=lambda item:item[0])
    if not np.allclose(R@R.T,np.eye(k),atol=1e-7) or not np.isclose(np.linalg.det(R),1.,atol=1e-7):
        raise RuntimeError("Optimizer returned non-SO(k) transformation")
    return R,dict(metric=name,loss=value,identity_loss=float(bank.score(np.eye(k),name)),
                  evaluations=evaluations,starts=starts,maxiter=maxiter,
                  success=bool(fit.success),message=str(fit.message))


def run_candidates(bank,config):
    names=list(config["candidates"])
    if not names or any(name not in ALL_OBJECTIVES for name in names) or len(names)!=len(set(names)):
        raise ValueError("candidates must be nonempty, unique, and in ALL_OBJECTIVES")
    primary=str(config["primary"])
    if primary not in names:
        raise ValueError("primary must occur in candidates")
    output={"identity":np.eye(bank.k)}
    reports={"identity":dict(metric="identity",loss=None,identity_loss=None,success=True)}
    for name in names:
        R,report=optimize_one(bank,name,config)
        output[name]=R
        reports[name]=report
    return output,reports
