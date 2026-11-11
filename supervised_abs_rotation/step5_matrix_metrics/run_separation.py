"""Wave J: functional separation-rank tails, screening CR3 first."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

import run as wave
import run_finite
import fitted_bank
import kernel_anova as ka
import kernel_separation as ks


def analyze(source, fields, output, seed=389, anchors_count=8):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output: {output}")
    fitted = fitted_bank.FittedBank(source, fields)
    data, old = fitted.data, fitted.protocol
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=old["model"], Z_dim=data.d, n_subjects=data.n,
        source=str(Path(source).resolve()), source_run_name=Path(fields).name,
        source_code_sha256=old["code_sha256"], seed=seed, estimators=old["estimators"],
        channels=list(data.channels), anchors=anchors_count, sigmas=[.2, .5],
        quadrature_nodes=12, spectral_energy_floor=1e-8,
        relative_L2_errors=[.01, .03, .1], effective_support_p=[.5, 1.],
        minimum_reference_coverage=.9, support_k=16, support_factor=1.5,
        all_subjects_used=True, empirical_input_independence_assumed=False,
        score_sign="lower for all datasets", status="RUNNING",
        code_sha256=hashlib.sha256(b"".join((wave.HERE / f).read_bytes() for f in
            ("run_separation.py", "kernel_separation.py", "fitted_bank.py", "kernel_anova.py", "tail_support.py", "matrix_io.py", "surrogates.py", "structure.py"))).hexdigest())
    wave.json_file(output / "protocol.json", config)
    rng = np.random.default_rng(seed)
    anchor_ids = rng.choice(data.n, min(data.n, anchors_count), replace=False)
    paths = [data.paths[i] for i in rng.permutation(len(data.paths))]
    maps, reveal = [], {}
    for k, path in enumerate(paths):
        q, _ = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); reveal[f"case_{k:03d}"] = path.parent.name
    blind, audits = [], []
    for name in old["estimators"]:
        model = fitted.model(name)
        anchors = model["z"][anchor_ids]
        tree = cKDTree(model["z"])
        radius = tree.query(anchors, k=min(16, data.n))[0][:, -1]
        coverage = {}
        for sigma in (.2, .5):
            noise = np.random.default_rng(seed+1).normal(size=(len(anchors), 128, data.d))
            probes = (anchors[:, None]+sigma*noise).reshape(-1, data.d)
            shifted = tree.query(probes, k=min(16, data.n))[0][:, -1].reshape(len(anchors), -1)
            coverage[sigma] = float(np.mean(shifted <= 1.5*radius[:, None]))
        if name.startswith("poly"):
            probes = np.random.default_rng(seed+2).normal(size=(4*len(model["powers"]), data.d))
        for k, q in enumerate(maps):
            if name.startswith("poly"):
                transform = ka.rotation_representation(model["powers"], model["weights"], q, probes)
                grams = [transform @ g @ transform.T for g in model["grams"]]
            for c, channel in enumerate(data.channels):
                for sigma in (.2, .5):
                    if name.startswith("poly"):
                        scores, audit = ks.polynomial_scores(model["powers"], model["weights"], grams[c], anchors @ q, sigma)
                    else:
                        scores, audit = ks.rbf_scores(model["centers"] @ q, model["grams"][c], anchors @ q,
                                                      model["lengthscale"], sigma)
                    audits.append(dict(anonymous_case=f"case_{k:03d}", estimator=name, channel=channel,
                                       sigma=sigma, reference_coverage=coverage[sigma], **audit))
                    for metric, loss in scores.items():
                        blind.append(dict(anonymous_case=f"case_{k:03d}", estimator=name, channel=channel,
                                          metric=f"s{sigma:g}_{metric}", loss=loss))
            if (k+1) % 8 == 0 or k == len(maps)-1:
                print(f"[WaveJ] {name}: {k+1}/{len(maps)} anonymous cases", flush=True)
        wave.csv_file(output / "blind_scores.csv", blind)
        wave.json_file(output / "separation_audit.json", audits)
    digest = hashlib.sha256((output / "blind_scores.csv").read_bytes()).hexdigest()
    wave.json_file(output / "reveal.json", dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row["estimator"], row["channel"], row["metric"]][reveal[row["anonymous_case"]]] = row["loss"]
    summaries = []
    for (name, channel, metric), values in grouped.items():
        sigma = float(metric.split("_")[0][1:])
        aa = [a for a in audits if a["estimator"] == name and a["channel"] == channel and a["sigma"] == sigma]
        qr = fitted.quality[name, channel]
        numerical = (max(a["quadrature_relative_error"] for a in aa) <= 1e-5 and
                     max(a["negative_relative_energy"] for a in aa) <= 1e-6)
        adequate = qr["adequate"] == "True" and numerical and min(a["reference_coverage"] for a in aa) >= .9
        summaries.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index["cases"]), adequate=bool(adequate),
            validation_nmse=qr["validation_nmse"], numerical_adequacy=numerical,
            reference_coverage=aa[0]["reference_coverage"], is_control=False))
    wave.csv_file(output / "metric_summary.csv", summaries)
    wave.json_file(output / "read_audit.json", dict(files=sorted(set(data.read_files)),
        derived_files=sorted(fitted.derived_reads), source_signature_sha256=fitted.source_digest,
        anchor_subjects=anchor_ids.tolist(), forbidden_files_read=[], all_subjects=data.n, all_rows=data.m))
    config.update(status="COMPLETE", scores_sha256=digest)
    wave.json_file(output / "protocol.json", config)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=wave.io.DIMS, default="CR3")
    p.add_argument("--run-name", default="waveJ_separation_seed389")
    p.add_argument("--source-run", default="waveC_finite_seed383")
    p.add_argument("--seed", type=int, default=389)
    p.add_argument("--anchors", type=int, default=8)
    p.add_argument("--batch", action="store_true")
    args = p.parse_args(); paths = {}
    with threadpool_limits(limits=4):
        for model in (("CR3", "C1", "C2", "C3") if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model)
            fields = source.parent / "step5_matrix_metrics" / args.source_run
            if not (fields / "protocol.json").exists():
                run_finite.analyze(source, fields)
            output = source.parent / "step5_matrix_metrics" / args.run_name
            paths[model] = output
            rows = analyze(source, fields, output, args.seed, args.anchors)
            passing = defaultdict(set)
            for r in rows:
                if r["adequate"] and r["all_planes_pass"]:
                    passing[r["channel"], r["metric"]].add("poly" if r["estimator"].startswith("poly") else "rbf")
            robust = [key for key, families in passing.items() if families >= {"poly", "rbf"}]
            print(f"[WaveJ] {model}: {len(robust)} cross-family spectral support candidates", flush=True)
            if args.batch and model == "CR3" and not robust:
                print("[WaveJ] CR3 failed; later models not launched", flush=True)
                return
        if args.batch:
            wave.compare(paths, wave.ROOT / "outputs" / "supervised_abs_rotation" / f"step5_{args.run_name}_four_models.csv")


if __name__ == "__main__":
    main()
