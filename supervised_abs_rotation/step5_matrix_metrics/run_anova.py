"""Wave F: exact local-reference functional ANOVA of WY-fitted functions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

import run as wave
import run_finite
import kernel_anova as ka


def analyze(source, fields, output, seed=386, anchors_count=16):
    source, fields, output = Path(source), Path(fields), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Retain old experiments; choose a new output: {output}")
    old = json.loads((fields / "protocol.json").read_text())
    if old["status"] != "COMPLETE" or hashlib.sha256((fields / "blind_scores.csv").read_bytes()).hexdigest() != old["scores_sha256"]:
        raise ValueError("Source score experiment incomplete or changed")
    data = wave.io.MatrixCases(source, tuple(old["channels"]), old["seed"])
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index["model"], Z_dim=data.d, n_subjects=data.n,
        source=str(source.resolve()), seed=seed, source_run_name=fields.name,
        source_code_sha256=old["code_sha256"], estimators=old["estimators"],
        channels=list(data.channels), anchors=anchors_count, sigmas=[.2, .5],
        reference="equal mixture of anchor-specific isotropic Gaussian product measures",
        minimum_reference_coverage=.9, support_k=16, support_factor=1.5,
        all_subjects_used=True, empirical_input_independence_assumed=False,
        score_sign="lower for all datasets", status="RUNNING",
        code_sha256=hashlib.sha256(b"".join((wave.HERE / f).read_bytes() for f in
            ("run_anova.py", "kernel_anova.py", "matrix_io.py", "surrogates.py", "structure.py"))).hexdigest())
    wave.json_file(output / "protocol.json", config)
    old_rng = np.random.default_rng(old["seed"])
    train = np.sort(old_rng.permutation(data.n)[:int(.75*data.n)])
    mu = data.u[train].mean(0)
    scale = np.sqrt(np.var(data.u[train], axis=0).mean())
    u = (data.u-mu)/scale
    rng = np.random.default_rng(seed)
    anchor_ids = rng.choice(data.n, min(anchors_count, data.n), replace=False)
    paths = [data.paths[i] for i in rng.permutation(len(data.paths))]
    maps, reveal = [], {}
    for k, path in enumerate(paths):
        q, _ = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); reveal[f"case_{k:03d}"] = path.parent.name
    with np.load(fields / "response_bank.npz", allow_pickle=False) as saved:
        bank = saved["responses"]
    variance = np.maximum(np.var(bank[train], axis=0), 1e-12)
    with (fields / "fit_quality.csv").open(newline="") as stream:
        quality = {(r["estimator"], r["channel"]): r for r in csv.DictReader(stream)}
    blind, audits, coverages = [], [], {}
    for name in old["estimators"]:
        with np.load(fields / f"model_{name}.npz", allow_pickle=False) as saved:
            coef = saved["coef"].copy()
            scaler = object.__new__(wave.sur.Scaler)
            scaler.mean = saved["mean"].copy(); scaler.scale = saved["scale"].copy()
            center_points = saved["centers"].copy()
            stimuli = saved["score_stimuli"].copy()
        z = scaler(u)
        anchors = z[anchor_ids]
        tree = cKDTree(z)
        radius = tree.query(anchors, k=min(16, data.n))[0][:, -1]
        for sigma in (.2, .5):
            noise = np.random.default_rng(seed+1).normal(size=(len(anchors), 128, data.d))
            probes = (anchors[:, None]+sigma*noise).reshape(-1, data.d)
            shifted_radius = tree.query(probes, k=min(16, data.n))[0][:, -1].reshape(len(anchors), -1)
            coverages[name, sigma] = float(np.mean(shifted_radius <= 1.5*radius[:, None]))
        if name.startswith("poly"):
            powers = wave.sur.powers_for(data.d, int(name[4:]))
            weights = wave.sur.weights_for(powers)
            coordinate_probes = np.random.default_rng(seed+2).normal(size=(4*len(powers), data.d))
            grams = []
            for c in range(2):
                normalized = coef[:, c*bank.shape[2]:(c+1)*bank.shape[2]]/np.sqrt(variance[c])[None, :]
                grams.append(normalized @ normalized.T)
        else:
            model = wave.sur.RBF(scaler, center_points, float(name[3:]))
            model.set_coef(coef)
            with np.load(fields / f"field_{name}.npz", allow_pickle=False) as saved:
                subjects = saved["subjects"][:8]
                expected = saved["gradient"][:8]
            gg, _ = model.derivatives(u[subjects])
            flat_ids = np.r_[stimuli, bank.shape[2]+stimuli]
            reconstructed = gg[:, flat_ids].reshape(expected.shape)
            if np.max(np.abs(reconstructed-expected)) > 1e-6*max(1., np.max(np.abs(expected))):
                raise ArithmeticError("Reconstructed Nystrom weights disagree with saved fields")
            grams = []
            for c in range(2):
                normalized = model.beta[:, c*bank.shape[2]:(c+1)*bank.shape[2]]/np.sqrt(variance[c])[None, :]
                grams.append(normalized @ normalized.T)
        for k, q in enumerate(maps):
            if name.startswith("poly"):
                transform = ka.rotation_representation(powers, weights, q, coordinate_probes)
                rotated_grams = [transform @ gram @ transform.T for gram in grams]
            for c, channel in enumerate(data.channels):
                for sigma in (.2, .5):
                    if name.startswith("poly"):
                        scores, audit = ka.polynomial_scores(powers, weights, rotated_grams[c], anchors @ q, sigma)
                    else:
                        scores, audit = ka.scores(model.centers @ q, grams[c], anchors @ q, model.lengthscale, sigma)
                    audits.append(dict(anonymous_case=f"case_{k:03d}", estimator=name, channel=channel,
                                       sigma=sigma, reference_coverage=coverages[name, sigma], **audit))
                    for metric, loss in scores.items():
                        blind.append(dict(anonymous_case=f"case_{k:03d}", estimator=name,
                            channel=channel, metric=f"s{sigma:g}_{metric}", loss=loss))
            if (k+1) % 8 == 0 or k == len(maps)-1:
                print(f"[WaveF] {name}: {k+1}/{len(maps)} anonymous cases", flush=True)
        wave.csv_file(output / "blind_scores.csv", blind)
        wave.json_file(output / "anova_audit.json", audits)
    digest = hashlib.sha256((output / "blind_scores.csv").read_bytes()).hexdigest()
    wave.json_file(output / "reveal.json", dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row["estimator"], row["channel"], row["metric"]][reveal[row["anonymous_case"]]] = row["loss"]
    summaries = []
    for (name, channel, metric), values in grouped.items():
        sigma = float(metric.split("_")[0][1:])
        qr = quality[name, channel]
        aa = [a for a in audits if a["estimator"] == name and a["channel"] == channel and a["sigma"] == sigma]
        numerical = (max(a["sum_relative_error"] for a in aa) <= 1e-5 and
                     max(a["negative_relative_energy"] for a in aa) <= 1e-7)
        control = grouped[name, channel, f"s{sigma:g}_control_total_local_variance"]
        drift = np.ptp(list(control.values()))/max(max(np.abs(list(control.values()))), 1e-24)
        adequate = qr["adequate"] == "True" and coverages[name, sigma] >= .9 and numerical and drift <= 1e-6
        summaries.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index["cases"]), adequate=bool(adequate),
            validation_nmse=qr["validation_nmse"], reference_coverage=coverages[name, sigma],
            control_relative_drift=drift, numerical_adequacy=numerical,
            is_control="_control_" in metric))
    wave.csv_file(output / "metric_summary.csv", summaries)
    wave.json_file(output / "read_audit.json", dict(files=sorted(set(data.read_files)),
        source_scores_sha256=old["scores_sha256"], forbidden_files_read=[],
        anchor_subject_ids=anchor_ids.tolist(), all_subjects=data.n, all_rows=data.m))
    config.update(status="COMPLETE", scores_sha256=digest)
    wave.json_file(output / "protocol.json", config)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=wave.io.DIMS, default="CR3")
    p.add_argument("--run-name", default="waveF_anova_seed386")
    p.add_argument("--source-run", default="waveC_finite_seed383")
    p.add_argument("--seed", type=int, default=386)
    p.add_argument("--anchors", type=int, default=16)
    p.add_argument("--batch", action="store_true")
    args = p.parse_args()
    paths = {}
    with threadpool_limits(limits=4):
        for model in (("CR3", "C1", "C2", "C3") if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model)
            fields = source.parent / "step5_matrix_metrics" / args.source_run
            if not (fields / "protocol.json").exists():
                run_finite.analyze(source, fields)
            output = source.parent / "step5_matrix_metrics" / args.run_name
            paths[model] = output
            rows = analyze(source, fields, output, args.seed, args.anchors)
            passed = [r for r in rows if r["adequate"] and r["all_planes_pass"] and not r["is_control"]]
            print(f"[WaveF] {model}: {len(passed)} adequate full-plane candidates", flush=True)
            if args.batch and model == "CR3" and not passed:
                print("[WaveF] CR3 failed; later models not launched", flush=True)
                return
        if args.batch:
            wave.compare(paths, wave.ROOT / "outputs" / "supervised_abs_rotation" / f"step5_{args.run_name}_four_models.csv")


if __name__ == "__main__":
    main()
