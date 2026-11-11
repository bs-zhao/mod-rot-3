"""Wave L: direct joint W/Y polynomial-coefficient support; CR3 first."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import fast_projection as fp
import coefficient_support as cs


def analyze(source, output, seed=391, degrees=(2, 3, 4), chunk=8192, device="auto"):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Retain old experiments; choose a new output: {output}")
    data = wave.io.MatrixCases(source, ("logit_nn", "p_nn"), seed)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index["model"], Z_dim=data.d, n_subjects=data.n,
        source=str(source.resolve()), seed=seed, degrees=list(degrees), channels=list(data.channels),
        chunk=chunk, device=device, heldout_fraction=.25, ridge_grid=[1e-6, 1e-4, .01],
        adequacy_validation_nmse=.1, all_subjects_used=True,
        descriptor_scope="support of isotropic joint U/X polynomial projections; no true-function-form assumption",
        score_sign="lower for all datasets", status="RUNNING",
        code_sha256=hashlib.sha256(b"".join((wave.HERE / f).read_bytes() for f in
            ("run_joint_projection.py", "fast_projection.py", "coefficient_support.py", "tail_support.py", "kernel_anova.py", "matrix_io.py", "surrogates.py", "structure.py"))).hexdigest())
    wave.json_file(output / "protocol.json", config)
    rng = np.random.default_rng(seed)
    train = np.sort(rng.permutation(data.n)[:int(.75*data.n)])
    scaler = wave.sur.Scaler(data.w, data.d, row_mask=np.isin(data.sid, train))
    maps, reveal, residuals = [], {}, []
    for k, path in enumerate(data.paths):
        q, residual = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); residuals.append(residual); reveal[f"case_{k:03d}"] = path.parent.name
    blind, quality = [], []
    for degree in degrees:
        model = fp.Polynomial(scaler, degree)
        print(f"[WaveL] {model.name}: fitting all {data.m:,} rows / {data.n:,} subjects", flush=True)
        coef, fit = fp.fit(model, data.w, data.y, data.sid, train, chunk=chunk, device=device)
        wave.json_file(output / f"fit_{model.name}.json", fit)
        np.savez_compressed(output / f"projection_{model.name}.npz", coef=coef, powers=model.powers,
            weights=model.weights, mean=scaler.mean, scale=scaler.scale, train_subjects=train)
        if "numeric_backend" in config and config["numeric_backend"] != fit["backend"]:
            raise ValueError("All degrees must use the same numeric backend")
        config["numeric_backend"] = fit["backend"]
        for c, channel in enumerate(data.channels):
            quality.append(dict(estimator=model.name, channel=channel, validation_nmse=fit["validation_nmse"][c],
                adequate=fit["validation_nmse"][c] <= .1, ridge=fit["ridge"], feature_count=len(model.powers),
                final_fit_subjects=data.n, final_fit_rows=data.m, backend=fit["backend"]))
        structure = cs.Structure(model.powers, data.d, seed)
        for k, q in enumerate(maps):
            rotated = structure.rotate(coef, q)
            for c, channel in enumerate(data.channels):
                for metric, loss in structure.scores(rotated[:, c]).items():
                    blind.append(dict(anonymous_case=f"case_{k:03d}", estimator=model.name,
                                      channel=channel, metric=metric, loss=loss))
        wave.csv_file(output / "blind_scores.csv", blind)
        wave.csv_file(output / "fit_quality.csv", quality)
        print(f"[WaveL] {model.name}: anonymous support scores saved; NMSE {fit['validation_nmse']}", flush=True)
    digest = hashlib.sha256((output / "blind_scores.csv").read_bytes()).hexdigest()
    wave.json_file(output / "reveal.json", dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row["estimator"], row["channel"], row["metric"]][reveal[row["anonymous_case"]]] = row["loss"]
    quality_map = {(r["estimator"], r["channel"]): r for r in quality}
    summaries = []
    for (name, channel, metric), values in grouped.items():
        control = metric.startswith("control_")
        if control and np.ptp(list(values.values())) > 1e-7*max(1., max(values.values())):
            raise ArithmeticError("Coefficient norm changed under an orthogonal coordinate transformation")
        qr = quality_map[name, channel]
        summaries.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index["cases"]), adequate=qr["adequate"],
            validation_nmse=qr["validation_nmse"], is_control=control))
    wave.csv_file(output / "metric_summary.csv", summaries)
    wave.json_file(output / "read_audit.json", dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_subjects=data.n, all_trial_rows=data.m, registration_max_residual=max(residuals)))
    config.update(status="COMPLETE", scores_sha256=digest)
    wave.json_file(output / "protocol.json", config)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=wave.io.DIMS, default="CR3")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--run-name", default="waveL_joint_projection_seed391")
    p.add_argument("--seed", type=int, default=391)
    p.add_argument("--degrees", default="2,3,4")
    p.add_argument("--chunk", type=int, default=8192)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu", "numpy"))
    args = p.parse_args()
    degrees = tuple(map(int, args.degrees.split(",")))
    if not degrees or min(degrees) < 2 or len(set(degrees)) != len(degrees) or args.chunk < 1:
        p.error("Distinct degrees >=2 and positive chunk required")
    paths, survivors = {}, None
    with threadpool_limits(limits=4):
        for model in (("CR3", "C1", "C2", "C3") if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model)
            output = source.parent / "step5_matrix_metrics" / args.run_name
            paths[model] = output
            rows = analyze(source, output, args.seed, degrees, args.chunk, args.device)
            passed = {(r["estimator"], r["channel"], r["metric"]) for r in rows
                      if r["adequate"] and r["all_planes_pass"] and not r["is_control"]}
            survivors = passed if survivors is None else survivors & passed
            print(f"[WaveL] {model}: {len(passed)} projection support passes; {len(survivors)} common survivors", flush=True)
            if args.batch and not survivors:
                print("[WaveL] No identical candidate remains; later models not launched", flush=True)
                return
        if args.batch:
            destination = wave.ROOT / "outputs" / "supervised_abs_rotation" / f"step5_{args.run_name}_four_models.csv"
            wave.compare(paths, destination)
            print(f"[WaveL] All-four low-order projection candidates: {len(survivors)}; validation still required", flush=True)


if __name__ == "__main__":
    main()
