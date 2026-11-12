"""Wave AR: GP relevance directly from matrix responses, with no U teacher."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import ux_predictive_complexity as uc
import conditional_ux as conditional
import cp_regression as cp
import gp_ard_complexity as gp
import conditional_x_witness as witness
import ux_metric_scope as scope


def analyze(source, output, bank_folder, centers=128, training_steps=30, refit_steps=60,
            starts=3, model_chunk=8, seed=416, device='cuda', X_contrast='omission', bank_factory=profiles.MatrixBank):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous direct-matrix GP experiment: {output}')
    bank = bank_factory(source, bank_folder); data = bank.data
    output.mkdir(parents=True, exist_ok=True)
    shared = bank.quality['mode'] == 'numerically verified shared observed X'
    protocol = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(Path(source).resolve()),
        seed=seed, channels=data.channels, source_matrix_manifest_sha256=bank.protocol['manifest_sha256'],
        response_reference=bank.quality['mode'], exact_original_response_grid=shared,
        global_output_transform=bank.protocol.get('response_transform', 'identity'),
        heldout_interpretation=bank.quality.get('validation_interpretation', 'original untransformed response functions'),
        U_function_approximation_before_GP=False, original_rows_in_response_preparation=data.m,
        centers=centers, training_steps=training_steps, all_subject_refit_steps=refit_steps, starts=starts,
        model_chunk=model_chunk, device=device, precision='float64', GP_adequacy_relative_mse=.01,
        output_mode_relative_squared_tolerance=1e-6, discarded_modes_included_in_fidelity_check=True,
        primary_scope='per-X normalized GP inverse-lengthscale relevance of direct response-matrix coupling',
        X_contrast=X_contrast, X_omission_degree=2 if X_contrast == 'omission' else None,
        score_sign='lower for every dataset', optimization_stability_verified=False,
        status='RUNNING', code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_matrix_gp_ard.py', 'matrix_function_profiles.py', 'gp_ard_complexity.py', 'cp_regression.py',
             'conditional_ux.py', 'ux_predictive_complexity.py', 'run_finite.py', 'additive_profiles.py',
             'matrix_io.py', 'local_support.py', 'tail_support.py', 'structure.py', 'conditional_x_witness.py',
             'ux_metric_scope.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    maps, reveal, errors = [], {}, []
    for k, path in enumerate(data.paths):
        q, error = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); errors.append(error); reveal[f'case_{k:03d}'] = path.parent.name
    source_train = np.sort(np.random.default_rng(383).permutation(data.n)[:int(.75*data.n)])
    if X_contrast == 'omission':
        operators, audit = uc.omission_operators(bank.x, wave.sur.Scaler(data.w[:, data.d:], 1).scale,
            bank.responses, source_train, seed=405, degrees=(2,))
        operators = operators['omit_d2']
    elif X_contrast == 'conditional_witness':
        operators, audit = witness.operators(bank.x)
    else:
        raise ValueError(f'Unknown X contrast: {X_contrast}')
    wave.json_file(output/'X_crossfit_audit.json', audit)
    for j, operator in enumerate(operators):
        np.save(output/f'X_contrast_column{j}.npy', operator, allow_pickle=False)
    tasks, targets, discarded, mode_audits = profiles.task_modes(bank.responses, operators)
    wave.json_file(output/'matrix_function_modes.json', dict(tasks=mode_audits,
        source_response_quality=bank.quality, exact_original_response_grid=shared))
    z = data.u-data.u.mean(0); z /= np.sqrt(np.var(data.u, axis=0).mean())
    classes, membership = cp.canonical_classes(z, maps)
    u = np.einsum('nd,gde->gne', z, classes)
    width = max(t.shape[1] for t in targets); values = np.zeros((len(tasks), data.n, width))
    for t, value in enumerate(targets):
        values[t, :, :value.shape[1]] = value
    retained = np.sum(values*values, axis=-1)
    rng = np.random.default_rng(seed); train = np.sort(rng.permutation(data.n)[:int(.75*data.n)])
    valid = np.setdiff1d(np.arange(data.n), train)
    center_ids = rng.choice(train, min(centers, len(train)), replace=False)
    np.savez_compressed(output/'data_geometry.npz', W_only_classes=classes, case_membership=membership,
        U_mean=data.u.mean(0), U_scale=np.sqrt(np.var(data.u, axis=0).mean()),
        training=train, validation=valid, center_subject_ids=center_ids,
        retained_energy_by_subject=retained, discarded_energy_by_subject=discarded)
    convert = lambda a, dtype=torch.float64: torch.as_tensor(a, dtype=dtype, device=device)
    model = gp.Model(convert(u), convert(values), center_ids, starts)
    training, validation_rows = convert(train, torch.long), convert(valid, torch.long)
    all_rows = torch.arange(data.n, device=device)
    print(f'[WaveAR] {data.index["model"]}: {len(model.eta)} independent direct-matrix GPs, '
          f'{width} output modes, exact shared grid={shared}', flush=True)
    def callback(phase):
        def update(row, history):
            wave.json_file(output/f'{phase}_optimization.json', dict(history=history,
                optimized_subjects=len(train) if phase == 'training' else data.n))
            torch.save(dict(state=model.snapshot(), phase=phase, step=row['step']), output/'latest_checkpoint.pt')
            print(f'[WaveAR] {phase} step {row["step"]}: '
                  f'best evidence loss/subject {row["best_mean_nll_per_subject"]:.6g}', flush=True)
        return update
    gp.optimize(model, training, training_steps, model_chunk, callback('training'))
    validation = np.empty(len(model.eta))
    for a in range(0, len(model.eta), model_chunk):
        ids = torch.arange(a, min(a+model_chunk, len(model.eta)), device=device)
        errors_retained = model.prediction_error(ids, training, validation_rows).cpu().numpy()
        target_ids = model.task_index[ids].cpu().numpy()
        validation[a:a+len(ids)] = profiles.full_error(errors_retained,
            retained[target_ids][:, valid].sum(1), discarded[target_ids][:, valid].sum(1))
    loss, _ = gp.optimize(model, all_rows, refit_steps, model_chunk, callback('all_subject_refit'))
    fit_error = np.empty(len(model.eta))
    for a in range(0, len(model.eta), model_chunk):
        ids = torch.arange(a, min(a+model_chunk, len(model.eta)), device=device)
        errors_retained = model.prediction_error(ids, all_rows, all_rows).cpu().numpy()
        target_ids = model.task_index[ids].cpu().numpy()
        fit_error[a:a+len(ids)] = profiles.full_error(errors_retained, retained[target_ids].sum(1), discarded[target_ids].sum(1))
    shape = (starts, len(tasks), len(classes)); loss = loss.reshape(shape)
    validation, fit_error = validation.reshape(shape), fit_error.reshape(shape)
    eta = model.eta.detach().cpu().numpy().reshape(*shape, data.d)
    np.savez_compressed(output/'GP_hyperparameters.npz', inverse_lengthscales=eta,
        signal_amplitude=np.exp(model.log_amplitude.detach().cpu().numpy().reshape(shape)),
        noise_sd=np.exp(model.log_noise.detach().cpu().numpy().reshape(shape)),
        evidence_loss=loss, validation_relative_mse=validation, all_subject_relative_mse=fit_error)
    selected = np.argmin(loss, axis=0)
    graph = np.zeros((len(classes), len(data.channels), data.d, data.dx))
    worst_valid = np.zeros(len(data.channels)); worst_fit = worst_valid.copy(); models = []
    for t, (c, j) in enumerate(tasks):
        for group in range(len(classes)):
            start = int(selected[t, group])
            if mode_audits[t]['rank']:
                graph[group, c, :, j] = eta[start, t, group]**2
            worst_valid[c] = max(worst_valid[c], validation[start, t, group])
            worst_fit[c] = max(worst_fit[c], fit_error[start, t, group])
            models.append(dict(channel=data.channels[c], X_column=j, coordinate_class=group,
                selected_start=start, validation_relative_mse=validation[start, t, group],
                all_subject_relative_mse=fit_error[start, t, group], evidence_loss=loss[start, t, group]))
    blind, quality = [], []
    for c, channel in enumerate(data.channels):
        quality.append(dict(estimator='matrix_GP_ARD', channel=channel,
            adequate=bank.quality['validation_nmse'][c] <= .1 and worst_valid[c] <= .01 and worst_fit[c] <= .01,
            validation_nmse=bank.quality['validation_nmse'][c], GP_worst_subject_validation_relative_mse=worst_valid[c],
            GP_worst_all_subject_relative_mse=worst_fit[c], exact_original_response_grid=shared))
        for k, group in enumerate(membership):
            for metric, value in conditional.sparsity(graph[group, c]).items():
                blind.append(dict(anonymous_case=f'case_{k:03d}', estimator='matrix_GP_ARD', channel=channel,
                                  metric='GP_ARD_'+metric, loss=value))
    wave.csv_file(output/'blind_scores.csv', blind); wave.csv_file(output/'fit_quality.csv', quality)
    wave.csv_file(output/'GP_model_quality.csv', models)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict); qm = {r['channel']: r for r in quality}
    for row in blind:
        grouped[row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = [dict(estimator='matrix_GP_ARD', channel=c, metric=m, **wave.st.evaluate_curve(v, data.index['cases']),
        adequate=qm[c]['adequate'], validation_nmse=qm[c]['validation_nmse'], is_control=not scope.primary(m),
        optimization_stability_verified=False) for (c, m), v in grouped.items()]
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        matrix_bank_manifest_sha256=bank.protocol['manifest_sha256'], all_subjects_in_GP_refit=data.n,
        original_rows_in_response_preparation=data.m, original_U_model_coefficients_read=False,
        registration_max_residual=max(errors)))
    protocol.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', protocol)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3'); p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveAR_matrix_GP_ARD_UX_seed416')
    p.add_argument('--centers', type=int, default=128); p.add_argument('--training-steps', type=int, default=30)
    p.add_argument('--refit-steps', type=int, default=60); p.add_argument('--starts', type=int, default=3)
    p.add_argument('--model-chunk', type=int, default=8); p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--X-contrast', choices=('omission', 'conditional_witness'), default='omission')
    args = p.parse_args(); survivors = None
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            folder = parent/'matrix_response_table_seed383'
            if not folder.exists():
                profiles.prepare(source, folder)
            rows = analyze(source, parent/args.run_name, folder, centers=args.centers,
                training_steps=args.training_steps, refit_steps=args.refit_steps, starts=args.starts,
                model_chunk=args.model_chunk, device=args.device, X_contrast=args.X_contrast)
            passed = {(r['channel'], r['metric']) for r in rows if r['adequate'] and r['all_planes_pass'] and not r['is_control']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveAR] {model}: {len(passed)} preliminary direct-matrix GP passes, {len(survivors)} common', flush=True)
            if args.batch and not survivors:
                print('[WaveAR] No identical criterion remains; optimizer and center-count stability are separate', flush=True)
                return


if __name__ == '__main__':
    main()
