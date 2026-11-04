"""Audited researcher acceptance of ONE frozen CR1 v4 s2 failure.

This is NOT a numerical relaxation of the oracle-recovery or
information-balance gates. The original report, with qualified=False and
two failed checks, remains immutable. A named review decision permits
exploratory NP/rotation work on the exact existing dataset and report.

Only the observed CR1 v4 5000x500 simulation seeded 20261006, with the
recorded numerical shortfall, can be accepted. Future data or versions
must run a new scientific review. Other models are never eligible.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ...utils.io import save_json
from ...utils.paths import stage_file

MODEL_ID="CR1"
VERSION="v4_shared_gamma_max500"
DATA_SEED=20261006
PRECHECK_SEED=20261006
ALLOWED_FAILED_CHECKS=frozenset({"oracle_recovery","information_balance"})
EXPECTED_PEARSON_R=np.array([
    .78870427, .89327526, .83100924, .93129171
],dtype=float)
EXPECTED_INFORMATION_RATIO=5.04203
MATCH_TOLERANCE=5e-5


def _fingerprint(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_frozen_review(model,report,manifest):
    """Validate exact observed limitations, not a generalizable cutoff."""
    if (getattr(model,"model_id",None)!=MODEL_ID
        or getattr(model,"model_version",None)!=VERSION
        or getattr(model,"n_subjects",None)!=5000
        or getattr(model,"trial_count_min",None)!=500
        or getattr(model,"trial_count_max",None)!=500
        or getattr(model,"n_factors",None)!=4):
        raise RuntimeError("Researcher acceptance applies only to frozen CR1 v4 5000x500")
    if (report.get("model_id")!=MODEL_ID
        or report.get("model_version")!=VERSION
        or report.get("precheck_seed")!=PRECHECK_SEED
        or report.get("qualified") is not False):
        raise RuntimeError("CR1 exception requires its original failed v4 precheck report")
    if (manifest.get("data_seed")!=DATA_SEED
        or manifest.get("model",{}).get("model_id")!=MODEL_ID
        or manifest.get("model",{}).get("model_version")!=VERSION
        or manifest.get("model",{}).get("trial_count_min")!=500
        or manifest.get("model",{}).get("trial_count_max")!=500):
        raise RuntimeError("Data manifest does not match the frozen CR1 v4 dataset")
    checks=report.get("checks",{})
    failed={key for key,value in checks.items()
            if not bool(value.get("passed",False))}
    if failed!=ALLOWED_FAILED_CHECKS:
        raise RuntimeError(
            f"CR1 reviewed acceptance covers exactly "
            f"{sorted(ALLOWED_FAILED_CHECKS)}, got {sorted(failed)}"
        )
    recovery=checks["oracle_recovery"]
    balance=checks["information_balance"]
    if recovery.get("acceptance_criterion")!="per_parameter_pearson_r":
        raise RuntimeError("The recorded CR1 oracle must use Pearson r")
    r=np.asarray(recovery.get("pearson_r"),dtype=float)
    ratio=float(balance.get("max_min_ratio",float("nan")))
    if (r.shape!=(4,) or not np.all(np.isfinite(r))
        or not np.allclose(r,EXPECTED_PEARSON_R,rtol=0,atol=MATCH_TOLERANCE)
        or not np.isfinite(ratio)
        or abs(ratio-EXPECTED_INFORMATION_RATIO)>MATCH_TOLERANCE):
        raise RuntimeError(
            "CR1 observed s2 metrics differ from the explicitly "
            "reviewed v4 numerical exception. No generic waiver is allowed."
        )
    if (not np.all(r[1:]>=.80)
        or not (.78<r[0]<.80)
        or not (4.0<ratio<5.1)):
        raise RuntimeError("CR1 recovery/information values changed since review")
    return {
        "research_decision":"accept_for_exploratory_NP_and_rotation",
        "scope":"one_existing_CR1_v4_5000x500_dataset_only",
        "disclosure":"ORIGINAL PRECHECK FAILED; this is NOT qualified=True",
        "model_id":MODEL_ID,
        "model_version":VERSION,
        "data_seed":DATA_SEED,
        "precheck_seed":PRECHECK_SEED,
        "original_qualified":False,
        "accepted_failures":sorted(failed),
        "original_pearson_r":r.tolist(),
        "original_pearson_threshold":.80,
        "alpha_shortfall":float(.80-r[0]),
        "information_balance_ratio":ratio,
        "original_information_balance_limit":4.,
        "reason":(
            "Researcher explicitly accepts alpha r=.78870427 (0.01130 "
            "below .80) and Fisher diagonal balance ratio=5.04203 "
            "(above 4) for exploratory CR1 v4 NP/rotation analysis, "
            "with full documentation and without representing s2 as passed."
        ),
        "all_other_checks_passed":True,
    }


def _paths(model):
    return {
        "data":stage_file(model,"s1_generate","dataset",".npz"),
        "manifest":stage_file(model,"s1_generate","manifest",".json"),
        "s2_report":stage_file(model,"s2_precheck","report",".json"),
    }


def _load(model):
    paths=_paths(model)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen CR1 v4 artifact: {path}")
    manifest=json.loads(paths["manifest"].read_text(encoding="utf-8"))
    report=json.loads(paths["s2_report"].read_text(encoding="utf-8"))
    decision=evaluate_frozen_review(model,report,manifest)
    return paths,decision


def approval_path(model):
    return stage_file(model,"s2_precheck","researcher_acceptance",".json")


def authorize_frozen_review(model):
    """Write the reproducible, dataset-bound record when explicitly requested."""
    paths,decision=_load(model)
    decision["artifact_sha256"]={
        name:_fingerprint(path) for name,path in paths.items()
    }
    destination=approval_path(model)
    save_json(decision,destination)
    print("RESEARCHER ACCEPTED TWO FAILED CR1 V4 CHECKS (NOT PASSED): "
          +str(destination),flush=True)
    return decision


def require_frozen_review(model):
    """Refuse a missing/stale/altered researcher acceptance certificate."""
    approval=approval_path(model)
    if not approval.exists():
        raise RuntimeError(
            "Original CR1 s2 failed. To explicitly accept this specific "
            "v4 dataset, use run_cr1_full.py --from-stage s3 "
            "--accept-reviewed-v4; never edit s2 report to qualified=True."
        )
    recorded=json.loads(approval.read_text(encoding="utf-8"))
    paths,decision=_load(model)
    for key,value in decision.items():
        if recorded.get(key)!=value:
            raise RuntimeError(
                f"CR1 reviewed acceptance record disagrees on {key!r}"
            )
    expected=recorded.get("artifact_sha256",{})
    if set(expected)!=set(paths):
        raise RuntimeError("CR1 reviewed acceptance is missing file hashes")
    for key,path in paths.items():
        if expected[key]!=_fingerprint(path):
            raise RuntimeError(
                f"CR1 s1/s2 artifact changed since researcher review: {key}. "
                "Refuse to reuse this acceptance for a different dataset."
            )
    print(
        "CR1 V4 PROCEEDING UNDER RESEARCHER REVIEWED EXCEPTION: "
        "original s2 qualified=False, alpha r=0.78870, "
        "Fisher information ratio=5.04203; other checks passed.",
        flush=True,
    )
    return recorded
