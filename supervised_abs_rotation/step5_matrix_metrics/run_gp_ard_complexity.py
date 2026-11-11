"""Wave AQ: GP ARD relevance support for observed-X coupling contrasts."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import load_npz
from threadpoolctl import threadpool_limits

import run as wave
import fitted_bank
import prepare_response_bank
import ux_predictive_complexity as uc
import conditional_ux as conditional
import cp_regression as cp
import gp_ard_complexity as gp


def analyze(source, output, bank_folder, estimators=('poly5', 'rbf2'), centers=128,
            training_steps=30, refit_steps=60, starts=3, model_chunk=8, seed=416, device='cuda'):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous GP ARD experiment: {output}')
    bank = fitted_bank.FittedBank(source, bank_folder); data = bank.data
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(Path(source).resolve()),
        seed=seed, source_manifest_sha256=bank.source_digest, estimators=estimators, channels=data.channels,
        centers=centers, training_steps=training_steps, all_subject_refit_steps=refit_steps,
        starts=starts, model_chunk=model_chunk, device=device, precision='float64',
        kernel='Nystrom ARD RBF, center jitter 1e-6', inverse_lengthscale_bounds=[0., 10.],
        signal_amplitude_bounds=[.05, 10.], noise_sd_bounds=[.001, 1.],
        objective='mean scalar GP negative log evidence over unit-energy X-contrast function ensemble',
        primary_scope='normalized support of inverse GP lengthscales in the Ui-by-Xj relevance graph',
        target_mode_relative_squared_tolerance=1e-6, source_latent_relative_squared_tolerance=1e-7,
        GP_adequacy_relative_mse=.01,
        X_mode='omit_d2', empirical_input_independence_assumed=False,
        all_subjects_used_in_final_optimization=True, all_original_rows_in_source_fit=data.m,
        score_sign='lower for every dataset', optimization_stability_verified=False, status='RUNNING',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_gp_ard_complexity.py', 'gp_ard_complexity.py', 'conditional_ux.py', 'ux_predictive_complexity.py',
             'cp_regression.py', 'fitted_bank.py', 'matrix_io.py', 'tail_support.py', 'surrogates.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    maps, reveal, registration_errors = [], {}, []
    for k, path in enumerate(data.paths):
        q, error = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); registration_errors.append(error); reveal[f'case_{k:03d}'] = path.parent.name
    cached = output.parent/'waveZ_UX_complexity_seed405'
    if cached.exists():
        old = json.loads((cached/'protocol.json').read_text())
        digest = hashlib.sha256((cached/'blind_scores.csv').read_bytes()).hexdigest()
        if old['status'] != 'COMPLETE' or digest != old['scores_sha256'] or old['source_signature_sha256'] != bank.source_digest:
            raise ValueError('X contrasts do not match this immutable function bank')
        operators = {'omit_d2': [load_npz(cached/f'X_omit_d2_column{j}.npz').toarray() for j in range(data.dx)]}
        wave.json_file(output/'X_map_audit.json', dict(source=str(cached.resolve()), scores_sha256=digest))
    else:
        with np.load(bank.folder/'response_bank.npz', allow_pickle=False) as saved:
            x = saved['x']
        operators, audit = uc.omission_operators(x, wave.sur.Scaler(data.w[:, data.d:], 1).scale,
            bank.bank, bank.train, seed=405, degrees=(2,))
        wave.json_file(output/'X_map_audit.json', audit)
    rng = np.random.default_rng(seed)
    train = np.sort(rng.permutation(data.n)[:int(.75*data.n)])
    valid = np.setdiff1d(np.arange(data.n), train)
    center_ids = rng.choice(train, min(centers, len(train)), replace=False)
    np.savez_compressed(output/'subject_split_and_centers.npz', training=train, validation=valid, center_subject_ids=center_ids)
    blind, quality, detailed_quality = [], [], []
    for name in estimators:
        coefficients, z, meta = uc.source_coefficients(bank, name)
        latents, _, grams, source_audit = conditional.balanced_latents(conditional.source_features(z, meta), coefficients, operators)
        classes, membership = cp.canonical_classes(z, maps)
        tasks, targets, mode_audits = [], [], []
        for c, channel in enumerate(data.channels):
            for j, gram in enumerate(grams['omit_d2'][c]):
                target, audit = gp.target_modes(latents, gram)
                tasks.append((c, j)); targets.append(target)
                mode_audits.append(dict(channel=channel, X_column=j, **audit))
        width = max(t.shape[1] for t in targets)
        values = np.zeros((len(tasks), data.n, width))
        for t, value in enumerate(targets):
            values[t, :, :value.shape[1]] = value
        wave.json_file(output/f'{name}_function_mode_audit.json', dict(source=source_audit, tasks=mode_audits))
        u = np.einsum('nd,gde->gne', z, classes)
        convert = lambda a, dtype=torch.float64: torch.as_tensor(a, dtype=dtype, device=device)
        model = gp.Model(convert(u), convert(values), center_ids, starts)
        train_tensor, valid_tensor = convert(train, torch.long), convert(valid, torch.long)
        all_tensor = torch.arange(data.n, device=device)
        print(f'[WaveAQ] {data.index["model"]} {name}: {len(model.eta)} independent GP fits, '
              f'{len(classes)} W classes, {len(tasks)} X/channel tasks, {width} output modes', flush=True)
        def callback(phase):
            def update(row, history):
                wave.json_file(output/f'{name}_{phase}_optimization.json', dict(history=history,
                    optimized_subjects=len(train) if phase == 'training' else data.n))
                print(f'[WaveAQ] {name} {phase} step {row["step"]}: '
                      f'best evidence loss/subject {row["best_mean_nll_per_subject"]:.6g}', flush=True)
                torch.save(dict(state=model.snapshot(), phase=phase, step=row['step']), output/f'{name}_latest_checkpoint.pt')
            return update
        gp.optimize(model, train_tensor, training_steps, model_chunk, callback('training'))
        validation = np.empty(len(model.eta))
        for a in range(0, len(model.eta), model_chunk):
            ids = torch.arange(a, min(a+model_chunk, len(model.eta)), device=device)
            validation[a:a+len(ids)] = model.prediction_error(ids, train_tensor, valid_tensor).cpu().numpy()
        final_loss, history = gp.optimize(model, all_tensor, refit_steps, model_chunk, callback('all_subject_refit'))
        fit_error = np.empty(len(model.eta))
        for a in range(0, len(model.eta), model_chunk):
            ids = torch.arange(a, min(a+model_chunk, len(model.eta)), device=device)
            fit_error[a:a+len(ids)] = model.prediction_error(ids, all_tensor, all_tensor).cpu().numpy()
        shape = (starts, len(tasks), len(classes))
        loss = final_loss.reshape(shape); validation = validation.reshape(shape); fit_error = fit_error.reshape(shape)
        eta = model.eta.detach().cpu().numpy().reshape(*shape, data.d)
        amplitude = np.exp(model.log_amplitude.detach().cpu().numpy().reshape(shape))
        noise = np.exp(model.log_noise.detach().cpu().numpy().reshape(shape))
        np.savez_compressed(output/f'{name}_GP_hyperparameters.npz', inverse_lengthscales=eta,
            signal_amplitude=amplitude, noise_sd=noise, final_evidence_loss=loss,
            subject_validation_relative_mse=validation, all_subject_relative_mse=fit_error,
            W_only_classes=classes, case_membership=membership)
        selected = np.argmin(loss, axis=0)
        graph = np.zeros((len(classes), len(data.channels), data.d, data.dx))
        worst_valid = np.zeros(len(data.channels)); worst_fit = worst_valid.copy()
        for t, (c, j) in enumerate(tasks):
            for group in range(len(classes)):
                start = int(selected[t, group])
                graph[group, c, :, j] = eta[start, t, group]**2
                worst_valid[c] = max(worst_valid[c], validation[start, t, group])
                worst_fit[c] = max(worst_fit[c], fit_error[start, t, group])
                detailed_quality.append(dict(estimator=name, channel=data.channels[c], X_column=j,
                    coordinate_class=group, selected_start=start, validation_relative_mse=validation[start, t, group],
                    all_subject_relative_mse=fit_error[start, t, group], evidence_loss=loss[start, t, group],
                    minimum_inverse_lengthscale=float(eta[start, t, group].min()),
                    maximum_inverse_lengthscale=float(eta[start, t, group].max())))
        for c, channel in enumerate(data.channels):
            old = bank.quality[name, channel]
            quality.append(dict(estimator=name, channel=channel,
                adequate=old['adequate'] == 'True' and worst_valid[c] <= .01 and worst_fit[c] <= .01,
                validation_nmse=float(old['validation_nmse']), GP_worst_subject_validation_relative_mse=worst_valid[c],
                GP_worst_all_subject_relative_mse=worst_fit[c], optimization_stability_verified=False))
            for case, group in enumerate(membership):
                for metric, value in conditional.sparsity(graph[group, c]).items():
                    blind.append(dict(anonymous_case=f'case_{case:03d}', estimator=name, channel=channel,
                        metric='GP_ARD_'+metric, loss=value))
        wave.csv_file(output/'blind_scores.csv', blind); wave.csv_file(output/'fit_quality.csv', quality)
        wave.csv_file(output/'GP_model_quality.csv', detailed_quality)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict); qm = {(r['estimator'], r['channel']): r for r in quality}
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = [dict(estimator=e, channel=c, metric=m, **wave.st.evaluate_curve(v, data.index['cases']),
        adequate=qm[e, c]['adequate'], validation_nmse=qm[e, c]['validation_nmse'],
        optimization_stability_verified=False, is_control=False) for (e, c, m), v in grouped.items()]
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        derived_function_files=sorted(bank.derived_reads), all_subjects_in_GP_refit=data.n,
        all_original_rows_in_source_fit=data.m, registration_max_residual=max(registration_errors)))
    protocol.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', protocol)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3'); p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveAQ_GP_ARD_UX_seed416')
    p.add_argument('--centers', type=int, default=128); p.add_argument('--training-steps', type=int, default=30)
    p.add_argument('--refit-steps', type=int, default=60); p.add_argument('--starts', type=int, default=3)
    p.add_argument('--model-chunk', type=int, default=8); p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args(); survivors = None
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            folder = parent/'UX_complexity_source_seed383'
            if not folder.exists():
                prepare_response_bank.prepare(source, folder, estimators=('poly3', 'poly5', 'rbf1', 'rbf2'))
            rows = analyze(source, parent/args.run_name, folder, centers=args.centers,
                training_steps=args.training_steps, refit_steps=args.refit_steps, starts=args.starts,
                model_chunk=args.model_chunk, device=args.device)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows if r['adequate'] and r['all_planes_pass']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveAQ] {model}: {len(passed)} preliminary adequate GP-ARD passes, {len(survivors)} common', flush=True)
            if args.batch and not survivors:
                print('[WaveAQ] No identical criterion remains; optimizer/center-resolution audit is separate', flush=True)
                return


if __name__ == '__main__':
    main()
