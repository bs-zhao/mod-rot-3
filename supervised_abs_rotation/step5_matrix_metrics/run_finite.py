"""Wave C: fixed-stimulus conditional monotonicity and finite-scale effects."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import additive_profiles as ap
import finite_effects as fe


def response_bank(data, seed):
    counts = np.diff(data.offsets)
    shared = bool(np.all(counts == counts[0]))
    x = np.asarray(data.w[:counts[0], data.d:])
    if shared:
        for a, b in zip(data.offsets[:-1], data.offsets[1:]):
            if not np.array_equal(data.w[a:b, data.d:], x):
                shared = False
                break
    if shared:
        bank = data.y.reshape(data.n, counts[0], len(data.channels)).transpose(0, 2, 1)
        return bank, x, dict(mode="numerically verified shared observed X", validation_nmse=[0.]*len(data.channels),
                             all_subjects=data.n, all_rows=data.m)
    bank, x, audit = ap.x_profiles(data, seed=seed, probe_count=128)
    audit["mode"] = "generic X dictionary evaluated at pooled observed X probes"
    audit["common_support_verified"] = False
    return bank, x, audit


def hessian_scores(h):
    d = h.shape[-1]
    eu = np.mean(h*h, axis=(0, 1))
    ii, jj = np.triu_indices(d, 1)
    out = wave.st.concentration(eu[ii, jj], "UU_edges")
    out.update(wave.st.concentration(np.r_[np.diag(eu), 2*eu[ii, jj]], "UU_all"))
    out["axis_curvature_l1_over_fro"] = float(np.mean(np.sum(np.abs(np.diagonal(h, axis1=-2, axis2=-1)), axis=-1)) /
        max(np.sqrt(eu.sum()), 1e-24))
    out["control_UU_fro2"] = float(eu.sum())
    return out


def analyze(source, output, seed=383, estimators=("poly3", "poly5", "rbf1", "rbf2"), centers=512):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new output; preserve failed experiments: {output}")
    data = wave.io.MatrixCases(source, ("logit_nn", "p_nn"), seed)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index["model"], Z_dim=data.d, n_subjects=data.n,
        source=str(Path(source).resolve()), seed=seed, centers=centers,
        estimators=list(estimators), channels=list(data.channels),
        steps=[.2, .5], contexts=[8, 32], support_radius_k=16, support_factor=1.5,
        minimum_common_support=.5, score_stimuli=64, heldout_fraction=.25,
        adequacy_validation_nmse=.1, all_subjects_used=True,
        score_sign="lower for all datasets", derivative_accuracy_verified=False,
        status="RUNNING", code_sha256=hashlib.sha256(b"".join((wave.HERE / f).read_bytes() for f in
            ("run_finite.py", "finite_effects.py", "additive_profiles.py", "matrix_io.py", "surrogates.py", "structure.py"))).hexdigest())
    wave.json_file(output / "protocol.json", config)
    rng = np.random.default_rng(seed)
    train = np.sort(rng.permutation(data.n)[:int(.75*data.n)])
    mu = data.u[train].mean(0)
    scale = np.sqrt(np.var(data.u[train], axis=0).mean())
    u = (data.u-mu)/scale
    maps, reveal = [], {}
    for i, path in enumerate(data.paths):
        q, residual = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); reveal[f"case_{i:03d}"] = path.parent.name
    bank, x, profile_quality = response_bank(data, seed)
    wave.json_file(output / "response_quality.json", profile_quality)
    stim = np.sort(rng.choice(bank.shape[2], min(64, bank.shape[2]), replace=False))
    flat_ids = np.r_[stim, bank.shape[2]+stim]
    target = bank.reshape(data.n, -1)
    np.savez_compressed(output / "response_bank.npz", responses=bank, x=x, score_stimuli=stim)
    scaler = wave.sur.Scaler(u, data.d, row_mask=np.isin(np.arange(data.n), train))
    center_ids = rng.choice(train, min(centers, len(train)), replace=False)
    mask = fe.support_mask(u, maps)
    keep = np.flatnonzero(mask)
    wave.json_file(output / "support.json", dict(common_subjects=len(keep), total_subjects=data.n,
        fraction=float(mask.mean()), minimum_fraction=.5, selection_uses_Y=False))
    np.save(output / "common_support_subjects.npy", keep)
    if len(keep) < 20:
        raise ValueError("Insufficient common support for finite effects")
    print(f"[WaveC] common support: {len(keep)}/{data.n} subjects; fitting every subject", flush=True)
    # Contexts depend on observed coordinates only, never responses or labels.
    context_list = [fe.contexts((u @ q)[keep]) for q in maps]
    blind, quality = [], []
    for name in estimators:
        model = (wave.sur.Polynomial(scaler, int(name[4:])) if name.startswith("poly") else
                 wave.sur.RBF(scaler, u[center_ids], float(name[3:])))
        fit = wave.sur.fit_all_rows(model, u, target, np.arange(data.n), train, chunk=1024)
        nmse = np.asarray(fit["validation_nmse"]).reshape(2, bank.shape[2]).mean(1)
        wave.json_file(output / f"fit_{name}.json", fit)
        np.savez_compressed(output / f"model_{name}.npz", coef=model.coef, mean=scaler.mean,
                           scale=scaler.scale, centers=u[center_ids], score_stimuli=stim)
        for c, channel in enumerate(data.channels):
            quality.append(dict(estimator=name, channel=channel, validation_nmse=float(nmse[c]),
                profile_nmse=profile_quality["validation_nmse"][c], common_support=float(mask.mean()),
                adequate=bool(nmse[c] <= .1 and profile_quality["validation_nmse"][c] <= .1 and mask.mean() >= .5),
                final_fit_subjects=data.n, final_fit_trial_rows=data.m))
        print(f"[WaveC] {name}: held-out NMSE {nmse.tolist()}", flush=True)
        model.set_coef(model.coef[:, flat_ids].copy())
        g = np.empty((len(keep), 2*len(stim), data.d))
        h = np.empty((*g.shape, data.d))
        for a in range(0, len(keep), 256):
            gg, hh = model.derivatives(u[keep[a:a+256]])
            g[a:a+len(gg)], h[a:a+len(gg)] = gg, hh
        g = g.reshape(len(keep), 2, len(stim), data.d)
        h = h.reshape(len(keep), 2, len(stim), data.d, data.d)
        np.savez_compressed(output / f"field_{name}.npz", gradient=g, hessian=h, subjects=keep)
        cache = {}

        def finite(direction, step):
            nonzero = np.flatnonzero(np.abs(direction) > 1e-8)
            sign = 1 if direction[nonzero[0]] > 0 else -1
            direction = direction*sign
            key = (step, tuple(np.round(direction, 7)))
            if key not in cache:
                values = np.empty((len(keep), 2*len(stim)))
                for a in range(0, len(keep), 512):
                    z = u[keep[a:a+512]]
                    values[a:a+len(z)] = ((model.features(z+step/2*direction)-
                        model.features(z-step/2*direction)) @ model.coef)/step
                cache[key] = values.reshape(len(keep), 2, len(stim))
            return sign*cache[key]

        for k, q in enumerate(maps):
            rotated_g = g @ q
            rotated_h = np.einsum("ij,ncsjk,kl->ncsil", q.T, h, q, optimize=True)
            for c, channel in enumerate(data.channels):
                scores = fe.effect_scores(rotated_g[:, c], context_list[k])
                scores.update(fe.ux_contrast_scores(rotated_g[:, c], x[stim]))
                scores.update(hessian_scores(rotated_h[:, c]))
                scores["control_U_gradient_fro2"] = float(np.mean(np.sum(rotated_g[:, c]**2, axis=-1)))
                for metric, loss in scores.items():
                    blind.append(dict(anonymous_case=f"case_{k:03d}", estimator=name, channel=channel,
                                      metric=metric, loss=loss))
            for step in (.2, .5):
                effects = np.stack([finite(q[:, i], step) for i in range(data.d)], axis=-1)
                for c, channel in enumerate(data.channels):
                    scores = fe.effect_scores(effects[:, c], context_list[k])
                    scores.update(fe.ux_contrast_scores(effects[:, c], x[stim]))
                    for metric, loss in scores.items():
                        blind.append(dict(anonymous_case=f"case_{k:03d}", estimator=name,
                            channel=channel, metric=f"finite{step:g}_{metric}", loss=loss))
            if (k+1) % 8 == 0 or k == len(maps)-1:
                print(f"[WaveC] {name}: {k+1}/{len(maps)} anonymous cases", flush=True)
        wave.csv_file(output / "blind_scores.csv", blind)
        wave.csv_file(output / "fit_quality.csv", quality)
    digest = hashlib.sha256((output / "blind_scores.csv").read_bytes()).hexdigest()
    wave.json_file(output / "reveal.json", dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row["estimator"], row["channel"], row["metric"]][reveal[row["anonymous_case"]]] = row["loss"]
    quality_map = {(r["estimator"], r["channel"]): r for r in quality}
    summaries = []
    for (name, channel, metric), losses in grouped.items():
        qr = quality_map[name, channel]
        summaries.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(losses, data.index["cases"]), adequate=qr["adequate"],
            validation_nmse=qr["validation_nmse"], is_control=metric.startswith("control_")))
        if metric.startswith("control_") and np.ptp(list(losses.values())) > 1e-7*max(1., max(losses.values())):
            raise ArithmeticError(f"Invariant control changed: {name}/{channel}/{metric}")
    wave.csv_file(output / "metric_summary.csv", summaries)
    wave.json_file(output / "read_audit.json", dict(files=sorted(set(data.read_files)),
        forbidden_files_read=[], all_subjects=data.n, all_rows=data.m))
    config.update(status="COMPLETE", scores_sha256=digest)
    wave.json_file(output / "protocol.json", config)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=wave.io.DIMS, default="CR3")
    p.add_argument("--run-name", default="waveC_finite_seed383")
    p.add_argument("--seed", type=int, default=383)
    p.add_argument("--estimators", default="poly3,poly5,rbf1,rbf2")
    p.add_argument("--batch", action="store_true")
    p.add_argument("--centers", type=int, default=512)
    args = p.parse_args()
    paths = {}
    with threadpool_limits(limits=4):
        for model in (("CR3", "C1", "C2", "C3") if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model)
            output = source.parent / "step5_matrix_metrics" / args.run_name
            paths[model] = output
            rows = analyze(source, output, args.seed, tuple(args.estimators.split(",")), args.centers)
            passed = [r for r in rows if r["adequate"] and r["all_planes_pass"] and not r["is_control"]]
            print(f"[WaveC] {model}: {len(passed)} adequate full-plane candidates", flush=True)
            if args.batch and model == "CR3" and not passed:
                print("[WaveC] CR3 failed; later models not launched", flush=True)
                return
        if args.batch:
            wave.compare(paths, wave.ROOT / "outputs" / "supervised_abs_rotation" / f"step5_{args.run_name}_four_models.csv")


if __name__ == "__main__":
    main()
