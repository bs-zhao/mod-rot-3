# Adapted verbatim function bodies from user-supplied rot1_config.zip.
# Source hashes and method mapping are recorded in source_manifest.json.
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import SplineTransformer, StandardScaler
SEED=31


def canonical_sign(v: np.ndarray) -> np.ndarray:
    """Make a vector's sign deterministic without changing its direction."""
    v = np.asarray(v, float).copy()
    j = int(np.argmax(np.abs(v)))
    if v[j] < 0:
        v *= -1.0
    return v

def orthogonal_complement(unit_vector: np.ndarray) -> np.ndarray:
    """Rows form an orthonormal basis for the orthogonal complement."""
    v = np.asarray(unit_vector, float).reshape(1, -1)
    _, _, vh = np.linalg.svd(v, full_matrices=True)
    C = vh[1:, :]
    # Deterministic row signs for reproducibility.
    for i in range(C.shape[0]):
        C[i] = canonical_sign(C[i])
    return C

def theta_to_rotation(theta: np.ndarray, d: int) -> np.ndarray:
    """Unconstrained antisymmetric parameters -> SO(d) rotation."""
    if d <= 1:
        return np.eye(d)
    A = np.zeros((d, d), float)
    c = 0
    for i in range(d):
        for j in range(i + 1, d):
            A[i, j] = theta[c]
            A[j, i] = -theta[c]
            c += 1
    return expm(A)

def prepare_rows(B_perp: np.ndarray, row_min_frac: float):
    """Keep non-negligible functional-signature rows and assign robust weights.

    Orthogonal rotation preserves each row norm, so the keep mask and row
    weights can be fixed before optimization.

    sqrt(row energy) weighting downweights tiny/noisy rows without allowing the
    strongest row alone to dominate the criterion.
    """
    B_perp = np.asarray(B_perp, float)
    row_energy = np.sum(B_perp ** 2, axis=1)
    max_e = float(np.max(row_energy)) if len(row_energy) else 0.0
    cutoff = max(1e-14, float(row_min_frac) * max_e)
    keep = row_energy >= cutoff
    if not np.any(keep):
        raise RuntimeError("No functional-signature rows survived row filtering.")

    Bk = B_perp[keep]
    ek = row_energy[keep]
    w = np.sqrt(np.maximum(ek, 1e-16))
    w = w / np.sum(w)
    return Bk, keep, w, row_energy, cutoff

def entropy_and_coverage(
    Bk: np.ndarray,
    row_weights: np.ndarray,
    R: np.ndarray,
    coverage_tau: float,
    eps: float = 1e-12,
):
    """Return normalized row entropy and a soft minimum-coverage penalty.

    Row entropy:
        p_rj = a_rj^2 / sum_j a_rj^2
        H_r  = -sum_j p_rj log(p_rj) / log(d)

    H=0 only for a one-hot row.  In particular, (.7,.7,0,0) is penalized.

    Coverage:
        compute each column's share of total functional energy and penalize
        ONLY shares below coverage_tau.  This is not an equal-energy objective.
    """
    A = np.asarray(Bk, float) @ np.asarray(R, float).T
    d = A.shape[1]

    sq = A ** 2
    row_sum = np.maximum(np.sum(sq, axis=1, keepdims=True), eps)
    P = sq / row_sum

    if d <= 1:
        H = np.zeros(len(A), float)
    else:
        H = -np.sum(P * np.log(P + eps), axis=1) / np.log(float(d))
    entropy = float(np.sum(row_weights * H))

    col_energy = np.sum(sq, axis=0)
    col_share = col_energy / max(float(np.sum(col_energy)), eps)

    tau = float(coverage_tau)
    if tau <= 0:
        coverage_penalty = 0.0
    else:
        shortfall = np.maximum(0.0, tau - col_share) / tau
        coverage_penalty = float(np.mean(shortfall ** 2))

    return entropy, coverage_penalty, col_share, A, H

def optimize_entropy_rotation(
    B_perp: np.ndarray,
    n_starts: int,
    seed: int,
    row_min_frac: float,
    coverage_lambda: float,
    coverage_tau: float,
):
    d = B_perp.shape[1]
    Bk, keep, w, row_energy, cutoff = prepare_rows(
        B_perp, row_min_frac=row_min_frac
    )

    if d <= 1:
        R = np.eye(d)
        ent, cov, share, _, H = entropy_and_coverage(
            Bk, w, R, coverage_tau
        )
        return R, {
            "objective_before": ent + coverage_lambda * cov,
            "objective_after": ent + coverage_lambda * cov,
            "entropy_before": ent,
            "entropy_after": ent,
            "coverage_penalty_before": cov,
            "coverage_penalty_after": cov,
            "column_share_before": share.tolist(),
            "column_share_after": share.tolist(),
            "n_rows_total": int(len(B_perp)),
            "n_rows_kept": int(np.sum(keep)),
            "row_cutoff_energy": float(cutoff),
            "success": True,
            "message": "d<=1",
            "nfev": 0,
        }

    p = d * (d - 1) // 2
    rng = np.random.default_rng(seed)

    def objective(theta):
        R = theta_to_rotation(theta, d)
        ent, cov, _, _, _ = entropy_and_coverage(
            Bk, w, R, coverage_tau
        )
        return ent + float(coverage_lambda) * cov

    starts = [np.zeros(p)]
    starts.extend(
        rng.normal(0.0, 0.35, size=p)
        for _ in range(max(0, int(n_starts) - 1))
    )

    best = None
    for x0 in starts:
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": 3000, "gtol": 1e-8, "ftol": 1e-13},
        )
        if best is None or float(res.fun) < float(best.fun):
            best = res

    R = theta_to_rotation(best.x, d)

    ent0, cov0, share0, _, _ = entropy_and_coverage(
        Bk, w, np.eye(d), coverage_tau
    )
    ent1, cov1, share1, _, _ = entropy_and_coverage(
        Bk, w, R, coverage_tau
    )

    meta = {
        "objective_before": float(ent0 + coverage_lambda * cov0),
        "objective_after": float(ent1 + coverage_lambda * cov1),
        "entropy_before": float(ent0),
        "entropy_after": float(ent1),
        "coverage_penalty_before": float(cov0),
        "coverage_penalty_after": float(cov1),
        "column_share_before": share0.tolist(),
        "column_share_after": share1.tolist(),
        "n_rows_total": int(len(B_perp)),
        "n_rows_kept": int(np.sum(keep)),
        "row_min_frac": float(row_min_frac),
        "row_cutoff_energy": float(cutoff),
        "coverage_lambda": float(coverage_lambda),
        "coverage_tau": float(coverage_tau),
        "success": bool(best.success),
        "message": str(best.message),
        "nfev": int(best.nfev),
        "best_objective": float(best.fun),
    }
    return R, meta
