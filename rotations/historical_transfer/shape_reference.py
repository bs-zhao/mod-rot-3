# Adapted verbatim function bodies from user-supplied rot1_config.zip.
# Source hashes and method mapping are recorded in source_manifest.json.
from __future__ import annotations
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import SplineTransformer, StandardScaler
SEED=31
from .regression_reference import within_subject_center


def canonicalize_rows(S: np.ndarray) -> np.ndarray:
    """Deterministic row signs; does not change the represented rotation."""
    S = np.asarray(S, float).copy()
    for i in range(S.shape[0]):
        j = int(np.argmax(np.abs(S[i])))
        if S[i, j] < 0:
            S[i] *= -1.0
    return S

def fit_vector_additive_splines(
    X: np.ndarray,
    Y: np.ndarray,
    subj_idx: np.ndarray,
    n_knots: int = 6,
    grid_n: int = 301,
):
    """Fit Y ~ sum_j spline(X_j) + all pairwise standardized X products.

    The pairwise products act as nuisance/interaction regressors so that a
    multiplicative factor such as X3*X4 does not get spuriously assigned to a
    univariate X3 or X4 shape.

    Returns a function-level energy matrix and second-derivative curvature
    matrix for each varying X.
    """
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)

    # Remove subject-specific gradient offsets before studying X-dependent shape.
    Yc = within_subject_center(Y, subj_idx)

    x_sd = X.std(axis=0, ddof=0)
    varying = np.where(x_sd > 1e-8)[0]
    if len(varying) == 0:
        raise RuntimeError("No varying observed X columns.")

    blocks = []
    block_meta = []
    offset = 0

    # Each X gets its own spline transformer + scaler, retained for grid eval.
    for j in varying:
        spl = SplineTransformer(
            n_knots=int(n_knots),
            degree=3,
            include_bias=False,
        )
        Bj = spl.fit_transform(X[:, [j]])

        scaler = StandardScaler()
        Bjz = scaler.fit_transform(Bj)

        p = Bjz.shape[1]
        blocks.append(Bjz)
        block_meta.append(
            {
                "x_index": int(j),
                "start": int(offset),
                "stop": int(offset + p),
                "spline": spl,
                "scaler": scaler,
            }
        )
        offset += p

    # Pairwise raw-X products soak up simple interactions.
    Xv = X[:, varying]
    Xvz = StandardScaler().fit_transform(Xv)
    pair_names = []
    for a in range(len(varying)):
        for b in range(a + 1, len(varying)):
            prod = (Xvz[:, a] * Xvz[:, b])[:, None]
            blocks.append(prod)
            pair_names.append(
                f"x{int(varying[a])+1}:x{int(varying[b])+1}"
            )

    Phi = np.concatenate(blocks, axis=1)

    ridge = RidgeCV(
        alphas=np.logspace(-6, 3, 31),
        fit_intercept=False,
    )
    ridge.fit(Phi, Yc)

    # Multioutput RidgeCV coef_ = n_outputs x n_features.
    coef = np.asarray(ridge.coef_, float).T

    pred = ridge.predict(Phi)
    ss_res = np.sum((Yc - pred) ** 2, axis=0)
    ss_tot = np.sum((Yc - Yc.mean(axis=0)) ** 2, axis=0)
    fit_r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)

    per_x = {}
    for meta in block_meta:
        j = meta["x_index"]
        start, stop = meta["start"], meta["stop"]
        spl = meta["spline"]
        scaler = meta["scaler"]

        # Avoid extremely sparse finite-sample edges when estimating derivatives.
        lo, hi = np.quantile(X[:, j], [0.01, 0.99])
        if not np.isfinite(lo + hi) or hi <= lo:
            lo, hi = float(np.min(X[:, j])), float(np.max(X[:, j]))
        grid = np.linspace(float(lo), float(hi), int(grid_n))

        Bg = spl.transform(grid[:, None])
        Bgz = scaler.transform(Bg)

        # Partial vector-valued function for this X only.
        F = Bgz @ coef[start:stop, :]

        # Remove each axis's mean over this X range: we only want shape.
        F = F - F.mean(axis=0, keepdims=True)

        d1 = np.gradient(F, grid, axis=0, edge_order=2)
        d2 = np.gradient(d1, grid, axis=0, edge_order=2)

        # Functional operators in latent-axis space.
        energy = (F.T @ F) / len(grid)
        slope = (d1.T @ d1) / len(grid)
        curvature = (d2.T @ d2) / len(grid)

        # Numerical symmetry.
        energy = 0.5 * (energy + energy.T)
        slope = 0.5 * (slope + slope.T)
        curvature = 0.5 * (curvature + curvature.T)

        per_x[int(j)] = {
            "grid": grid,
            "F": F,
            "energy": energy,
            "slope": slope,
            "curvature": curvature,
        }

    return {
        "varying": varying,
        "per_x": per_x,
        "fit_r2": fit_r2,
        "ridge_alpha": float(ridge.alpha_),
        "pair_names": pair_names,
        "Yc_var": np.var(Yc, axis=0, ddof=0),
    }

def detect_shared_x_groups(
    fit,
    level_axis: int = 0,
    min_effect_fraction: float = 0.08,
    min_dominance: float = 0.55,
):
    """Assign non-level axes to a dominant univariate X, if one is substantial.

    An axis is assigned only if:
    - its strongest partial univariate functional energy is at least
      `min_effect_fraction` of its total gradient variance; and
    - that X accounts for at least `min_dominance` of its total univariate
      partial energy.

    Interaction-only axes can therefore remain unassigned.
    """
    per_x = fit["per_x"]
    yvar = np.maximum(np.asarray(fit["Yc_var"], float), 1e-12)
    K = len(yvar)

    x_keys = sorted(per_x.keys())
    diag_energy = np.vstack(
        [np.diag(per_x[j]["energy"]) for j in x_keys]
    )  # nX x K

    groups = {j: [] for j in x_keys}
    diagnostics = []

    for a in range(K):
        if a == int(level_axis):
            diagnostics.append(
                {
                    "axis_zero_based": int(a),
                    "assigned_x_zero_based": None,
                    "reason": "fixed_level_axis",
                }
            )
            continue

        vals = np.maximum(diag_energy[:, a], 0.0)
        total_uni = float(np.sum(vals))
        imax = int(np.argmax(vals))
        jstar = int(x_keys[imax])
        vmax = float(vals[imax])

        effect_fraction = vmax / float(yvar[a])
        dominance = vmax / max(total_uni, 1e-12)

        assigned = (
            effect_fraction >= float(min_effect_fraction)
            and dominance >= float(min_dominance)
        )

        if assigned:
            groups[jstar].append(int(a))

        diagnostics.append(
            {
                "axis_zero_based": int(a),
                "assigned_x_zero_based": int(jstar) if assigned else None,
                "max_univariate_energy": vmax,
                "total_univariate_energy": total_uni,
                "effect_fraction_of_gradient_variance": effect_fraction,
                "dominance_within_univariate_energy": dominance,
            }
        )

    groups = {j: axes for j, axes in groups.items() if len(axes) >= 2}
    return groups, diagnostics, diag_energy, x_keys

def curvature_refinement_rotation(
    K: int,
    groups: dict,
    fit: dict,
):
    """Build an orthogonal block rotation that diagonalizes curvature per group."""
    S = np.eye(K)
    group_info = []

    for xj, axes in sorted(groups.items()):
        axes = list(map(int, axes))
        Rfull = np.asarray(fit["per_x"][xj]["curvature"], float)
        Rg = Rfull[np.ix_(axes, axes)]

        # Symmetric eigendecomposition. Columns of V are eigenvectors.
        evals, V = np.linalg.eigh(0.5 * (Rg + Rg.T))
        order = np.argsort(evals)
        evals = np.maximum(evals[order], 0.0)
        V = V[:, order]

        # For u_new = Sg u_old, choose Sg = V.T so:
        # Sg Rg Sg.T = V.T Rg V = diagonal.
        Sg = canonicalize_rows(V.T)

        S[np.ix_(axes, axes)] = Sg

        R_after = Sg @ Rg @ Sg.T
        off_before = Rg - np.diag(np.diag(Rg))
        off_after = R_after - np.diag(np.diag(R_after))

        group_info.append(
            {
                "x_zero_based": int(xj),
                "axes_zero_based": axes,
                "curvature_eigenvalues": evals.tolist(),
                "curvature_matrix_before": Rg.tolist(),
                "curvature_matrix_after": R_after.tolist(),
                "offdiag_fro_before": float(np.linalg.norm(off_before)),
                "offdiag_fro_after": float(np.linalg.norm(off_after)),
                "group_rotation": Sg.tolist(),
            }
        )

    err = float(np.max(np.abs(S @ S.T - np.eye(K))))
    return S, group_info, err
