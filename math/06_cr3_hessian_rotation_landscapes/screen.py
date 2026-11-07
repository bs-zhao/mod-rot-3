"""Post-hoc strict selection of rotation-based Hessian sparsity losses.

An acceptable candidate has three properties on the SAME rotation plane:
(1) analytic CR3 vs NN-Y Hessian LOSS curves correlate positively;
(2) analytic CR3 vs analytic-Y-regression curves also correlate positively;
(3) the minimum of EACH curve is at 0° or 90°, up to a configurable
    angular tolerance; a maximum does not qualify.

Nearly flat signals are rejected. No analytic-Hessian data are used for
fitting the regression, only for post-hoc scientific evaluation.
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np

SOURCES=("analytic","formula_y","nn_y")
TARGETS=("choice","logit","probability")


def _is_loss_column(name):
    return name.endswith("_loss") or name in (
        "normalized_l1","offdiag_energy_ratio","row_entropy","effective_rank"
    )


def _pearson(x,y):
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)
    a=x-np.mean(x);b=y-np.mean(y)
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/norm) if norm>1e-14 else None


def _series_stats(series,angles,endpoint_tolerance,minimum_range):
    values=np.asarray(series,dtype=float)
    if not np.isfinite(values).all():return None
    magnitude=max(float(np.max(np.abs(values))),1e-12)
    variation=float(np.ptp(values)/magnitude)
    arg=int(np.argmin(values))
    min_angle=float(angles[arg])
    near_endpoint=bool(min(min_angle,90-min_angle)<=endpoint_tolerance+1e-9)
    original=float(values[0]);last=float(values[-1])
    # Endpoint agreement is a compulsory sanity check only for
    # coordinate permutation/sign invariant metrics. A number of
    # metrics may be numerically biased by the data estimator.
    return dict(minimum_angle=min_angle,minimum_value=float(values[arg]),
                value_at_zero=original,value_at_ninety=last,
                endpoint_delta=float(abs(original-last)),
                relative_variation=variation,
                informative=bool(variation>=minimum_range),
                near_endpoint_minimum=near_endpoint)


def select_sparsity_metrics(rows,*,min_correlation=.90,
                            endpoint_tolerance=10.,min_relative_range=.02):
    if not rows:return [],[],[]
    meta={"plane","parameter_i","parameter_j","angle_degrees","source","target","block"}
    # ZX-specific row losses are missing by design from ZZ/XX rows.
    # Use UNION, then check availability inside each (plane,Y,block) group.
    possible=set().union(*(row.keys() for row in rows))
    metric_names=sorted(k for k in possible if k not in meta and _is_loss_column(k))
    groups=defaultdict(lambda:defaultdict(dict))
    for row in rows:
        group=(row["plane"],row["target"],row["block"])
        groups[group][row["source"]][float(row["angle_degrees"])]=row
    checks=[]
    for (plane,target,block),by_source in groups.items():
        if not all(s in by_source for s in SOURCES):continue
        angles=np.array(sorted(by_source["analytic"]),dtype=float)
        if not np.isclose(angles[0],0) or not np.isclose(angles[-1],90):
            continue
        if any(list(sorted(by_source[source]))!=list(angles) for source in SOURCES):
            continue
        for key in metric_names:
            arrays={};stats={}
            valid=True
            for source in SOURCES:
                try:
                    values=[float(by_source[source][float(a)][key]) for a in angles]
                except (KeyError,TypeError,ValueError):
                    valid=False;break
                st=_series_stats(values,angles,endpoint_tolerance,min_relative_range)
                if st is None:valid=False;break
                arrays[source]=values;stats[source]=st
            if not valid:continue
            rnn=_pearson(arrays["analytic"],arrays["nn_y"])
            rformula=_pearson(arrays["analytic"],arrays["formula_y"])
            rformula_nn=_pearson(arrays["formula_y"],arrays["nn_y"])
            # Relative curve discrepancies are measured in units of the
            # ANALYTIC curve's observed angle variation; if that curve
            # is nearly flat, absolute errors are not diagnostic.
            denominator=max(float(np.ptp(arrays["analytic"])),1e-12)
            infer_rmse=float(np.sqrt(np.mean((
                np.asarray(arrays["formula_y"])-arrays["analytic"])**2))/denominator)
            additional_nn_rmse=float(np.sqrt(np.mean((
                np.asarray(arrays["nn_y"])-arrays["formula_y"])**2))/denominator)
            combined_nn_rmse=float(np.sqrt(np.mean((
                np.asarray(arrays["nn_y"])-arrays["analytic"])**2))/denominator)
            near_all=all(stats[s]["near_endpoint_minimum"] for s in SOURCES)
            varying_all=all(stats[s]["informative"] for s in SOURCES)
            strong=bool(rnn is not None and rformula is not None
                        and rformula_nn is not None
                        and rnn>=min_correlation and rformula>=min_correlation
                        and rformula_nn>=min_correlation
                        and near_all and varying_all)
            truth_end=bool(stats["analytic"]["near_endpoint_minimum"])
            truth_informative=bool(stats["analytic"]["informative"])
            formula_good=bool(rformula is not None
                              and rformula>=min_correlation
                              and stats["formula_y"]["near_endpoint_minimum"]
                              and stats["formula_y"]["informative"])
            nn_good=bool(rnn is not None
                         and rnn>=min_correlation
                         and stats["nn_y"]["near_endpoint_minimum"]
                         and stats["nn_y"]["informative"])
            if not truth_end or not truth_informative:
                diagnosis="analytic_H_not_endpoint_simple_or_uninformative"
            elif not formula_good and not nn_good:
                diagnosis=("shared_fit_bias_candidate" if rformula_nn is not None
                           and rformula_nn>=min_correlation
                           else "both_fitted_channels_fail_differently")
            elif not formula_good:
                diagnosis="formula_Y_fit_fails_even_if_NN_fit_looks_good"
            elif not nn_good:
                diagnosis="NN_Y_additional_discrepancy_candidate"
            else:
                diagnosis="three_way_support"
            checks.append(dict(
                plane=plane,target=target,block=block,metric=key,
                analytic_nn_curve_pearson=rnn,
                analytic_formula_curve_pearson=rformula,
                formula_nn_curve_pearson=rformula_nn,
                analytic_formula_curve_relative_rmse=infer_rmse,
                nn_formula_curve_relative_rmse=additional_nn_rmse,
                analytic_nn_curve_relative_rmse=combined_nn_rmse,
                formula_fit_meets_endpoint_and_correlation=formula_good,
                nn_fit_meets_endpoint_and_correlation=nn_good,
                failure_diagnosis=diagnosis,
                binary_choice_sampling_vs_hard_choice_caveat=target=="choice",
                analytic_min_angle=stats["analytic"]["minimum_angle"],
                formula_min_angle=stats["formula_y"]["minimum_angle"],
                nn_min_angle=stats["nn_y"]["minimum_angle"],
                analytic_relative_variation=stats["analytic"]["relative_variation"],
                formula_relative_variation=stats["formula_y"]["relative_variation"],
                nn_relative_variation=stats["nn_y"]["relative_variation"],
                analytic_endpoint_delta=stats["analytic"]["endpoint_delta"],
                nn_endpoint_delta=stats["nn_y"]["endpoint_delta"],
                all_three_endpoint_minima=near_all,
                all_three_informative=varying_all,
                qualifies=strong,
            ))
    per=defaultdict(list)
    for row in checks:
        per[(row["metric"],row["target"],row["block"])].append(row)
    detail=[]
    for (metric,target,block),entries in per.items():
        good=[r for r in entries if r["qualifies"]]
        rnn=[r["analytic_nn_curve_pearson"] for r in entries if r["analytic_nn_curve_pearson"] is not None]
        rf=[r["analytic_formula_curve_pearson"] for r in entries if r["analytic_formula_curve_pearson"] is not None]
        rn=[r["formula_nn_curve_pearson"] for r in entries if r["formula_nn_curve_pearson"] is not None]
        detail.append(dict(metric=metric,target=target,block=block,
                           planes_evaluated=len(entries),
                           planes_good=len(good),
                           qualifies_all_six_planes=bool(len(entries)==6 and len(good)==6),
                           mean_analytic_nn_pearson=float(np.mean(rnn)) if rnn else None,
                           mean_analytic_formula_pearson=float(np.mean(rf)) if rf else None,
                           mean_formula_nn_pearson=float(np.mean(rn)) if rn else None,
                           planes_formula_fit_bad=sum(not r["formula_fit_meets_endpoint_and_correlation"] for r in entries),
                           planes_nn_fit_bad=sum(not r["nn_fit_meets_endpoint_and_correlation"] for r in entries),
                           planes_shared_bias_candidate=sum(r["failure_diagnosis"]=="shared_fit_bias_candidate" for r in entries),
                           fraction_planes_good=len(good)/len(entries)))
    detail.sort(key=lambda row:(
        -row["planes_good"],-row["fraction_planes_good"],
        -(-1 if row["mean_analytic_nn_pearson"] is None else row["mean_analytic_nn_pearson"]),
        row["metric"],row["target"],row["block"]))
    totals=defaultdict(list)
    for row in detail:
        totals[row["metric"]].append(row)
    ranking=[]
    for metric,entries in totals.items():
        nn=[r["mean_analytic_nn_pearson"] for r in entries if r["mean_analytic_nn_pearson"] is not None]
        fm=[r["mean_analytic_formula_pearson"] for r in entries if r["mean_analytic_formula_pearson"] is not None]
        fn=[r["mean_formula_nn_pearson"] for r in entries if r["mean_formula_nn_pearson"] is not None]
        ranking.append(dict(metric=metric,
                            group_count=len(entries),
                            planes_passed_total=sum(r["planes_good"] for r in entries),
                            plane_tests_total=sum(r["planes_evaluated"] for r in entries),
                            max_planes_in_one_group=max(r["planes_good"] for r in entries),
                            full_six_plane_successes=sum(r["qualifies_all_six_planes"] for r in entries),
                            mean_analytic_nn_curve_corr=float(np.mean(nn)) if nn else None,
                            mean_analytic_formula_curve_corr=float(np.mean(fm)) if fm else None,
                            mean_formula_nn_curve_corr=float(np.mean(fn)) if fn else None,
                            total_planes_formula_fit_bad=sum(r["planes_formula_fit_bad"] for r in entries),
                            total_planes_shared_bias_candidate=sum(r["planes_shared_bias_candidate"] for r in entries)))
    ranking.sort(key=lambda row:(
        -row["full_six_plane_successes"],
        -row["max_planes_in_one_group"],
        -row["planes_passed_total"],
        -(row["mean_analytic_nn_curve_corr"] if row["mean_analytic_nn_curve_corr"] is not None else -1),
        row["metric"]))
    return checks,detail,ranking
