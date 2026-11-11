"""Inspect frozen real-data scores and retain per-plane failures and curves."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import textwrap
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
    if "UX_entries_point_mean_effective_support_p0p5" in available:
        metrics = ["UX_entries_subject_mean_soft_count_eps003",
                   "UX_columns_subject_mean_fractional_count_eps003",
                   "UU_edges_point_mean_effective_support_p0p5",
                   "UU_UX_total_point_median_fractional_count_eps001"]
    elif "UX_observed_contrast_point_mean_effective_support_p0p5" in available:
        metrics = ["UU_edges_point_mean_effective_support_p0p5",
                   "UU_all_subject_mean_fractional_count_eps003",
                   "UX_observed_contrast_point_mean_soft_count_eps003",
                   "U_gradient_context_mean_fractional_count_eps010"]
    elif "UX_entries_subject_mean_effective_support_p0p5" in available:
        metrics = ["UX_columns_subject_median_fractional_count_eps010",
                   "UX_entries_subject_mean_soft_count_eps003",
                   "UU_all_subject_mean_fractional_count_eps003",
                   "UU_edges_subject_mean_soft_count_eps001"]
    elif "s0.5_Uorder3_U_U_axis_subsets_fractional_count_eps001" in available:
        metrics = ["s0.5_Uorder3_U_U_axis_subsets_fractional_count_eps001",
                   "s0.5_equal_orders_U_U_axis_subsets_soft_count_eps001",
                   "s1_UX_coupling_UX_block_full_mean_effective_support_p0p5",
                   "s1_U_dependent_UX_block_full_mean_effective_support_p0p5"]
    elif "prediction_tail2_l2" in available:
        metrics = ["prediction_tail2_l2", "prediction_tail8_l2",
                   "prediction_count_eps010", "prediction_count_eps100"]
    elif "s0.5_UX_block_full_mean_effective_support_p0p5" in available:
        metrics = ["s0.5_UX_block_full_mean_effective_support_p0p5",
                   "s1_UX_block_full_mean_effective_support_p0p5",
                   "s0.5_UX_block_centered_mean_soft_count_eps003",
                   "s2_UX_block_centered_worst_fractional_count_eps010"]
    elif "all_U_effective_support_p0p5" in available:
        metrics = ["all_U_effective_support_p0p5", "UX_terms_effective_support_p0p5",
                   "UU_cross_terms_fractional_count_eps003", "UX_columns_soft_count_eps003"]
    elif "s0.2_axis_mean_effective_support_p0p5" in available:
        metrics = ["s0.2_axis_mean_effective_support_p0p5", "s0.5_axis_mean_effective_support_p0p5",
                   "s0.2_axis_mean_soft_count_eps003", "s0.5_axis_worst_fractional_count_eps010"]
    elif "s0.2_interaction_components_effective_support_p0p5" in available:
        metrics = ["s0.2_interaction_components_effective_support_p0p5", "s0.5_interaction_components_effective_support_p0p5",
                   "s0.2_interaction_components_soft_count_eps003", "s0.5_pair_components_fractional_count_eps010"]
    elif "global_UU_all_effective_support_p0p5" in available:
        metrics = ["global_UU_all_effective_support_p0p5", "context_UU_edges_effective_support_p0p5",
                   "point_UX_entries_soft_count_eps003", "context_UX_columns_fractional_count_eps010"]
    elif "s0.2_interaction_components_tail1_l2" in available:
        metrics = ["s0.2_interaction_components_tail1_l2", "s0.5_interaction_components_tail1_l2",
                   "s0.5_interaction_fraction", "s0.5_higher_order_fraction"]
    elif "l2_profiled_edge_fraction" in available:
        metrics = ["l2_profiled_edge_fraction", "l1_profiled_edge_fraction",
                   "l1_pooled_all_tail1_l2", "l2_point_edges_tail1_l2"]
    elif "unknown_link_test_nmse" in available:
        metrics = ["unknown_link_test_nmse", "identity_link_test_nmse",
                   "link_component_tail1_l2", "component_direction_variation"]
    elif "fixed_X_direction_inconsistency" in available:
        metrics = ["fixed_X_direction_inconsistency", "conditional_direction_k32",
                   "finite0.5_conditional_direction_k32", "UX_contrast_tail1_l2"]
    elif "additive_test_nmse" in available:
        metrics = ["additive_test_nmse", "pairwise_test_nmse", "pair_projection_tail1_l2", "component_tail1_l2"]
    else:
        metrics = ["UX_column_tail1_l2", "UU_edges_tail1_l2", "UU_interaction_fraction", "U_global_sign_inconsistency"]
    metrics = [metric for metric in metrics if metric in available]
    if not metrics:
        # New experiments must yield real curves rather than empty axes.
        # This is a display rule only; strict assessment uses every score.
        preferred = sorted(available, key=lambda metric: (
            metric.startswith('control_'), 'UX' not in metric,
            'columns_mean' not in metric, 'effective_support_p1' not in metric, metric))
        metrics = preferred[:4]
    estimators = sorted({r["estimator"] for r in curves})
    fig, axes = plt.subplots(len(metrics), len(estimators),
        figsize=(max(8, 4*len(estimators)), max(4, 2.8*len(metrics))), squeeze=False)
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
                ax.set_ylabel('\n'.join(textwrap.wrap(metric.replace('_', ' '), width=32))+
                              "\nloss minus baseline", fontsize=7)
            ax.set_xticks([0, 30, 60, 90])
    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        axes[0, -1].legend(handles, labels, fontsize=7, ncol=2)
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
