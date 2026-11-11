"""Inspect frozen real-data scores and retain per-plane failures and curves."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import run as wave


def inspect(folder, destination):
    folder, destination = Path(folder), Path(destination)
    protocol = json.loads((folder / "protocol.json").read_text())
    if protocol["status"] != "COMPLETE":
        raise ValueError("Only inspect completed, frozen scores")
    destination.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((folder / "reveal.json").read_text())
    digest = hashlib.sha256((folder / "blind_scores.csv").read_bytes()).hexdigest()
    if digest != protocol["scores_sha256"] or digest != frozen["scores_sha256"]:
        raise ValueError("Frozen numerical scores changed")
    reveal = frozen["mapping"]
    index = json.loads((Path(protocol["source"]) / "index.json").read_text())
    groups = defaultdict(dict)
    with (folder / "blind_scores.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            groups[row["estimator"], row["channel"], row["metric"]][reveal[row["anonymous_case"]]] = float(row["loss"])
    curves, planes = [], []
    for key, losses in groups.items():
        estimator, channel, metric = key
        base = losses["baseline"]
        span = float(np.ptp(list(losses.values())))
        by_plane = defaultdict(dict)
        for case in index["cases"]:
            if case.get("pair_1based"):
                pair = "-".join(map(str, case["pair_1based"]))
                by_plane[pair][int(case["angle_deg"])] = losses[case["id"]]
        for pair, values in by_plane.items():
            values[0] = base
            interior = [values[a] for a in (15, 30, 45, 60, 75)]
            margin = min(interior) - max(base, values[90])
            planes.append(dict(estimator=estimator, channel=channel, metric=metric,
                               plane=pair, min_interior_margin=margin,
                               margin_over_span=margin / max(span, 1e-24),
                               best_angle=min(values, key=values.get),
                               endpoint_error=abs(values[90] - base)))
            for angle, loss in sorted(values.items()):
                curves.append(dict(estimator=estimator, channel=channel, metric=metric,
                                   plane=pair, angle=angle, loss=loss,
                                   delta_from_baseline=loss-base))
    wave.csv_file(destination / "plane_diagnostics.csv", planes)
    wave.csv_file(destination / "plane_curves.csv", curves)
    with (folder / "metric_summary.csv").open(newline="") as stream:
        summaries = list(csv.DictReader(stream))
    adequate = [r for r in summaries if r["adequate"] == "True" and r["is_control"] == "False"]
    adequate.sort(key=lambda r: (-int(r["success_planes"]), int(r["baseline_rank"])))
    wave.json_file(destination / "assessment.json", dict(
        model=protocol["model"], complete=True,
        fully_passing=sum(r["all_planes_pass"] == "True" for r in adequate),
        best_adequate_candidates=adequate[:12]))
    print(json.dumps(dict(model=protocol["model"], passing=sum(r["all_planes_pass"] == "True" for r in adequate),
                          best=adequate[:8]), indent=2))
    return curves


def plot(curves, path, model, channel="logit_nn"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    available = {r["metric"] for r in curves}
    if "fixed_X_direction_inconsistency" in available:
        metrics = ["fixed_X_direction_inconsistency", "conditional_direction_k32",
                   "finite0.5_conditional_direction_k32", "UX_contrast_tail1_l2"]
    elif "additive_test_nmse" in available:
        metrics = ["additive_test_nmse", "pairwise_test_nmse", "pair_projection_tail1_l2", "component_tail1_l2"]
    else:
        metrics = ["UX_column_tail1_l2", "UU_edges_tail1_l2", "UU_interaction_fraction", "U_global_sign_inconsistency"]
    estimators = sorted({r["estimator"] for r in curves})
    fig, axes = plt.subplots(len(metrics), len(estimators), figsize=(18, 11), squeeze=False)
    for i, metric in enumerate(metrics):
        for j, estimator in enumerate(estimators):
            ax = axes[i, j]
            planes = sorted({r["plane"] for r in curves})
            for plane in planes:
                rows = [r for r in curves if r["metric"] == metric and r["estimator"] == estimator and r["channel"] == channel and r["plane"] == plane]
                if rows:
                    ax.plot([r["angle"] for r in rows], [r["delta_from_baseline"] for r in rows], marker=".", label=plane)
            ax.axhline(0, color="black", linewidth=.7)
            if i == 0:
                ax.set_title(estimator)
            if j == 0:
                ax.set_ylabel(metric + "\nloss minus baseline")
            ax.set_xticks([0, 30, 60, 90])
    axes[0, -1].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{model}: {channel}; positive interior values favor baseline")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=wave.io.DIMS)
    parser.add_argument("--run-name", default="waveA_seed381")
    args = parser.parse_args()
    folder = wave.io.default_step2(wave.ROOT, args.model).parent / "step5_matrix_metrics" / args.run_name
    dest = folder / "inspection"
    curves = inspect(folder, dest)
    plot(curves, dest / "representative_curves.png", args.model)


if __name__ == "__main__":
    main()
