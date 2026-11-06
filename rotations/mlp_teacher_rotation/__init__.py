"""Complete independent frozen-decoder soft-target rotation experiment suite."""
from __future__ import annotations
import numpy as np
from .teacher import build_teacher
from .surrogate import make_proxy
from .optimization import MetricBank, run_candidates

_LAST_RUN = None


def fit(source, config):
    """Accept only frozen FIT data and an S3 checkpoint, never theta."""
    global _LAST_RUN
    soft_y, direct, teacher_info = build_teacher(source,config)
    proxy = make_proxy(source["q_fit"],source["x_fit"],soft_y,
                       source["mask_fit"],config)
    bank = MetricBank(proxy,direct,config)
    rotations,optimizers = run_candidates(bank,config)
    name = str(config["primary"])
    R = rotations[name]
    if not np.allclose(R@R.T,np.eye(len(R)),atol=1e-7):
        raise RuntimeError("Nonorthogonal primary rotation")
    info = dict(teacher=teacher_info,
                response_profile_fidelity=proxy.fidelity,
                primary=name,candidate_names=list(rotations),
                optimizers=optimizers,
                identity_metrics=bank.diagnose(np.eye(len(R))),
                primary_metrics=bank.diagnose(R),
                no_theta_access="Frozen teacher and rotation candidates fit with TRAIN ZXY only")
    arrays = {
        "mlp_soft_y_fit":soft_y,
        "mlp_direct_probability_gradient":direct["probability"],
        "mlp_direct_logit_gradient":direct["logit"],
        "mlp_common_x":direct["common_x"],
        "mlp_anchor_fit_subject_ids":direct["anchor_fit_subject_ids"],
        "mlp_dz_dq":direct["dz_dq"],
    }
    arrays.update({f"polynomial_tensor_degree{degree}":a
                   for degree,a in proxy.coef.items()})
    _LAST_RUN = (bank,rotations,optimizers,teacher_info)
    return R,info,arrays


def postfit(folder,config):
    if _LAST_RUN is None:
        raise RuntimeError("No current suite fit; postfit cannot reconstruct prior in-memory state")
    from .postfit import save_diagnostics
    save_diagnostics(folder,*_LAST_RUN,config)
