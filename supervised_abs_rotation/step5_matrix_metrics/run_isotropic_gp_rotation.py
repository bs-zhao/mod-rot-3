"""BB: fit radial GP coupling functions once, then rotate their actual effects."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import conditional_x_witness as witness
import gp_ard_complexity as gp
import isotropic_gp_rotation as radial
import tree_rotation_stability as rotations
import ux_metric_scope as scope

PRIMARY = 'point_UX_columns_Xmean_subject_mean_soft_count_eps010'


def analyze(source, output, bank_folder, centers=128, training_steps=30, refit_steps=60,
            random_count=64, optimizer_starts=2, optimizer_evaluations=100, seed=416, device='cuda'):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous radial GP experiment: {output}')
    bank = profiles.MatrixBank(source, bank_folder); data = bank.data
    output.mkdir(parents=True)
    protocol = dict(status='RUNNING', model=data.index['model'], source=str(Path(source).resolve()),
        all_subjects=data.n, all_original_rows=data.m, source_matrix_manifest_sha256=bank.protocol['manifest_sha256'],
        source_response_reference=bank.quality['mode'], seed=seed, centers=centers,
        training_steps=training_steps, all_subject_refit_steps=refit_steps, starts=3,
        source_fit='one isotropic GP per X/channel task, exact common-eta constraint',
        coordinate_evaluation='rotate fixed posterior gradient covariance; no per-case refit',
        primary_blind_metric=PRIMARY, primary_blind_channel='logit_nn', random_SO_rotations=random_count,
        blind_optimizer_starts=optimizer_starts, blind_optimizer_evaluations=optimizer_evaluations,
        prediction_quality_policy='continuous diagnostic, no fixed veto',
        derivative_accuracy_verified=False, precision='float64', device=device,
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_isotropic_gp_rotation.py', 'isotropic_gp_rotation.py', 'gp_ard_complexity.py',
             'gp_posterior_effects.py', 'matrix_function_profiles.py', 'conditional_x_witness.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    operators, x_audit = witness.operators(bank.x)
    wave.json_file(output/'X_crossfit_audit.json', x_audit)
    for j, operator in enumerate(operators):
        np.save(output/f'X_contrast_column{j}.npy', operator, allow_pickle=False)
    tasks, targets, discarded, mode_audit = profiles.task_modes(bank.responses, operators)
    wave.json_file(output/'matrix_function_modes.json', dict(tasks=mode_audit))
    width = max(t.shape[1] for t in targets); values = np.zeros((len(tasks), data.n, width))
    for t, target in enumerate(targets):
        values[t, :, :target.shape[1]] = target
    retained = np.sum(values*values, axis=-1)
    z = data.u-data.u.mean(0); z /= np.sqrt(np.var(data.u, axis=0).mean())
    rng = np.random.default_rng(seed); train = np.sort(rng.permutation(data.n)[:int(.75*data.n)])
    valid = np.setdiff1d(np.arange(data.n), train)
    center_ids = rng.choice(train, min(centers, len(train)), replace=False)
    np.savez_compressed(output/'source_geometry.npz', U_mean=data.u.mean(0), U_scale=np.sqrt(np.var(data.u, axis=0).mean()),
        center_subject_ids=center_ids, training=train, validation=valid)
    convert = lambda a, dtype=torch.float64: torch.as_tensor(a, device=device, dtype=dtype)
    fitted = radial.model(convert(z[None]), convert(values), center_ids, starts=3)
    tr, va = convert(train, torch.long), convert(valid, torch.long)
    all_rows = torch.arange(data.n, device=device)
    def callback(phase):
        def log(row, history):
            wave.json_file(output/f'{phase}_optimization.json', dict(history=history))
            print(f'[WaveBB] {data.index["model"]} {phase} {row["step"]}: '
                  f'{row["best_mean_nll_per_subject"]:.5g}', flush=True)
        return log
    gp.optimize(fitted, tr, training_steps, chunk=6, callback=callback('training'))
    validation = []
    for a in range(0, len(fitted.eta), 6):
        ids = torch.arange(a, min(a+6, len(fitted.eta)), device=device)
        t = fitted.task_index[ids].cpu().numpy()
        error = fitted.prediction_error(ids, tr, va).cpu().numpy()
        validation.extend(profiles.full_error(error, retained[t][:, valid].sum(1), discarded[t][:, valid].sum(1)))
    losses, _ = gp.optimize(fitted, all_rows, refit_steps, chunk=6, callback=callback('refit'))
    selected = np.argmin(losses.reshape(3, len(tasks)), axis=0)
    ids = selected*len(tasks)+np.arange(len(tasks))
    covariance = np.zeros((data.n, len(data.channels), data.dx, data.d, data.d)); quality = []
    for a in range(0, len(tasks), 4):
        chosen = ids[a:a+4]
        field, error = radial.covariances(fitted, convert(chosen, torch.long), all_rows)
        for k, (c, j) in enumerate(tasks[a:a+len(chosen)]):
            t = a+k
            covariance[:, c, j] = field[k]
            full_error = profiles.full_error(error[k], retained[t].sum(), discarded[t].sum())
            quality.append(dict(channel=data.channels[c], X_column=j, selected_start=int(selected[t]),
                validation_relative_mse=float(validation[int(chosen[k])]), all_subject_relative_mse=float(full_error),
                original_profile_validation_nmse=bank.quality['validation_nmse'][c]))
    np.savez_compressed(output/'source_GP_parameters.npz', inverse_lengthscales=fitted.eta.detach().cpu().numpy(),
        signal_amplitude=np.exp(fitted.log_amplitude.detach().cpu().numpy()),
        noise_sd=np.exp(fitted.log_noise.detach().cpu().numpy()), selected_start=selected, evidence_loss=losses)
    np.savez_compressed(output/'all_subject_gradient_covariances.npz', covariance=covariance)
    wave.csv_file(output/'source_fit_quality.csv', quality)
    maps, reveal, blind = [], {}, []
    for k, path in enumerate(data.paths):
        q, _ = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); reveal[f'case_{k:03d}'] = path.parent.name
        for c, channel in enumerate(data.channels):
            for metric, value in radial.scores(covariance[:, c], q).items():
                blind.append(dict(anonymous_case=f'case_{k:03d}', estimator='radial_GP_posterior',
                    channel=channel, metric=metric, loss=value))
    wave.csv_file(output/'blind_scores.csv', blind)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    # The blind primary objective and starts do not read case labels.
    random_rng = np.random.default_rng(seed+10); random_q, random_loss = [], []
    for _ in range(random_count):
        q = rotations.random_rotation(data.d, random_rng)
        random_q.append(q); random_loss.append(radial.primary_value(covariance[:, 0], q))
    history, optimized = [], []
    for start, index in enumerate(np.argsort(random_loss)[:optimizer_starts]):
        q0 = random_q[int(index)]
        def objective(parameters):
            value = radial.primary_value(covariance[:, 0], q0 @ rotations.increment(parameters, data.d))
            history.append(dict(start=start, parameters=np.asarray(parameters).tolist(), loss=value))
            return value
        result = minimize(objective, np.zeros(data.d*(data.d-1)//2), method='Powell',
            bounds=[(-np.pi/3, np.pi/3)]*(data.d*(data.d-1)//2),
            options=dict(maxfev=optimizer_evaluations, xtol=.001, ftol=0.))
        best = min((r for r in history if r['start'] == start), key=lambda r:r['loss'])
        optimized.append(dict(loss=best['loss'], rotation=q0 @ rotations.increment(best['parameters'], data.d),
                              converged=bool(result.success)))
    wave.csv_file(output/'blind_optimization.csv', history)
    np.savez_compressed(output/'blind_rotations.npz', random=np.asarray(random_q), random_loss=np.asarray(random_loss),
        optimized=np.asarray([r['rotation'] for r in optimized]), optimized_loss=np.asarray([r['loss'] for r in optimized]))
    wave.json_file(output/'blind_freeze.json', dict(scores_sha256=digest,
        rotation_scores_sha256=hashlib.sha256((output/'blind_rotations.npz').read_bytes()).hexdigest()))
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = [dict(estimator='radial_GP_posterior', channel=c, metric=m,
        **wave.st.evaluate_curve(values, data.index['cases']), is_control=not scope.primary(m),
        prediction_error_veto_applied=False) for (c, m), values in grouped.items()]
    wave.csv_file(output/'metric_summary.csv', rows)
    baseline = next(float(r['loss']) for r in blind if r['channel']=='logit_nn' and r['metric']==PRIMARY
                    and reveal[r['anonymous_case']]=='baseline')
    assessment = dict(primary_metric=PRIMARY, baseline_loss=baseline,
        random_minimum_loss=float(min(random_loss)), blind_optimized_minimum_loss=min((r['loss'] for r in optimized), default=None),
        source_GP_refitted_per_rotation=False, source_prediction_errors_reported_without_cutoff=True)
    wave.json_file(output/'assessment.json', assessment)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[], all_subjects=data.n))
    protocol.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', protocol)
    print(f'[WaveBB] {data.index["model"]}: {assessment}', flush=True)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3'); p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveBB_radial_GP_fixed_function_seed416')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        for model_name in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model_name); parent = source.parent/'step5_matrix_metrics'
            folder = parent/'matrix_response_table_seed383'
            if not folder.exists():
                profiles.prepare(source, folder)
            analyze(source, parent/args.run_name, folder, device=args.device)
