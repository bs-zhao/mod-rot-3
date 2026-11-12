"""BA: UX sparsity of fitted GP function effects, not ARD hyperparameters."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import rank_response_profiles as ranked
import cp_regression as cp
import gp_ard_complexity as gp
import gp_posterior_effects as posterior
import ux_metric_scope as scope


def analyze(source, original, output, device='cuda', chunk=4):
    original, output = Path(original), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous posterior-effect wave: {output}')
    old = json.loads((original/'protocol.json').read_text())
    digest = hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()
    if old['status'] != 'COMPLETE' or digest != old['scores_sha256']:
        raise ValueError('Source GP fit must be complete with immutable scores')
    transform = old.get('global_output_transform', 'identity')
    folder = original.parent/('matrix_response_table_seed383' if transform == 'identity' else f'matrix_response_{transform}_seed383')
    bank = (profiles.MatrixBank if transform == 'identity' else ranked.MatrixBank)(source, folder); data = bank.data
    if old['source_matrix_manifest_sha256'] != bank.protocol['manifest_sha256']:
        raise ValueError('GP target source changed')
    operators = [np.load(original/f'X_contrast_column{j}.npy', allow_pickle=False) for j in range(data.dx)]
    tasks, targets, discarded, audits = profiles.task_modes(bank.responses, operators)
    with np.load(original/'data_geometry.npz', allow_pickle=False) as saved:
        classes, membership = saved['W_only_classes'], saved['case_membership']
        z = (data.u-saved['U_mean'])/saved['U_scale']
        center_ids = saved['center_subject_ids']
        original_retained = saved['retained_energy_by_subject']
        original_discarded = saved['discarded_energy_by_subject']
    maps = [wave.io.registration(data.u, data.coordinates(path))[0] for path in data.paths]
    verified, member_check = cp.canonical_classes(z, maps)
    if not np.array_equal(member_check, membership) or not np.allclose(verified, classes, atol=1e-10):
        raise ValueError('Original GP coordinate classes differ from actual W matrices')
    width = max(t.shape[1] for t in targets); values = np.zeros((len(tasks), data.n, width))
    for t, value in enumerate(targets):
        values[t, :, :value.shape[1]] = value
    retained = np.sum(values*values, axis=-1)
    if not np.allclose(retained, original_retained, atol=1e-10) or not np.allclose(discarded, original_discarded, atol=1e-10):
        raise ArithmeticError('Reconstructed GP function modes do not reproduce the source targets')
    with np.load(original/'GP_hyperparameters.npz', allow_pickle=False) as saved:
        eta = saved['inverse_lengthscales']; signal = saved['signal_amplitude']; noise = saved['noise_sd']
        loss = saved['evidence_loss']; expected_fit = saved['all_subject_relative_mse']
    output.mkdir(parents=True)
    protocol = dict(status='RUNNING', model=data.index['model'], all_subjects=data.n, source=str(Path(source).resolve()),
        GP_source_run=str(original.resolve()), GP_scores_sha256=digest, GP_refitted=False,
        source_matrix_manifest_sha256=bank.protocol['manifest_sha256'], global_output_transform=transform,
        effect_definitions=['posterior mean derivative', 'centered finite changes .05/.2 common U-scale units'],
        prediction_quality_policy='continuous diagnostic, no fixed veto', device=device, precision='float64',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_gp_posterior_support.py', 'gp_posterior_effects.py', 'gp_ard_complexity.py',
             'matrix_function_profiles.py', 'rank_response_profiles.py', 'local_support.py', 'conditional_ux.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    convert = lambda a, dtype=torch.float64: torch.as_tensor(a, device=device, dtype=dtype)
    model = gp.Model(convert(np.einsum('nd,gde->gne', z, classes)), convert(values), center_ids, starts=eta.shape[0])
    with torch.no_grad():
        model.eta.copy_(convert(eta.reshape(-1, data.d)))
        model.log_amplitude.copy_(convert(np.log(signal).ravel()))
        model.log_noise.copy_(convert(np.log(noise).ravel()))
    selected = np.argmin(loss, axis=0)
    ids = np.array([((int(selected[t, g])*len(tasks)+t)*len(classes)+g)
                    for t in range(len(tasks)) for g in range(len(classes))])
    locations = [(g, *tasks[t]) for t in range(len(tasks)) for g in range(len(classes))]
    energy_fields = {}; maximum_fidelity_error = 0.
    all_rows = torch.arange(data.n, device=device)
    for a in range(0, len(ids), chunk):
        batch_ids = ids[a:a+chunk]
        fields, retained_error = posterior.effects(model, convert(batch_ids, torch.long), all_rows)
        target_ids = model.task_index[convert(batch_ids, torch.long)].cpu().numpy()
        full_error = profiles.full_error(retained_error, retained[target_ids].sum(1), discarded[target_ids].sum(1))
        maximum_fidelity_error = max(maximum_fidelity_error, float(np.max(np.abs(full_error-expected_fit.ravel()[batch_ids]))))
        if maximum_fidelity_error > 1e-6:
            raise ArithmeticError('Posterior reconstruction does not reproduce the saved GP prediction error')
        for mode, energies in fields.items():
            if mode not in energy_fields:
                energy_fields[mode] = np.zeros((len(classes), len(data.channels), data.n, data.d, data.dx))
            for k, (g, c, j) in enumerate(locations[a:a+len(batch_ids)]):
                energy_fields[mode][g, c, :, :, j] = energies[k]
        if (a//chunk+1) % 8 == 0 or a+chunk >= len(ids):
            print(f'[WaveBA] {data.index["model"]}: posterior effects {min(a+chunk,len(ids))}/{len(ids)} selected GPs', flush=True)
    blind, by_class = [], {}
    for mode, field in energy_fields.items():
        np.savez_compressed(output/f'{mode}_all_subject_energies.npz', energy=field)
        for g in range(len(classes)):
            for c, channel in enumerate(data.channels):
                by_class[mode, g, c] = posterior.scores(field[g, c])
        for k, group in enumerate(membership):
            for c, channel in enumerate(data.channels):
                for metric, value in by_class[mode, int(group), c].items():
                    blind.append(dict(anonymous_case=f'case_{k:03d}', estimator='GP_posterior', channel=channel,
                                      metric=mode+'_'+metric, loss=value))
    wave.csv_file(output/'blind_scores.csv', blind)
    new_digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    reveal = {f'case_{k:03d}': path.parent.name for k, path in enumerate(data.paths)}
    wave.json_file(output/'reveal.json', dict(scores_sha256=new_digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = [dict(estimator='GP_posterior', channel=c, metric=m, **wave.st.evaluate_curve(v, data.index['cases']),
                 is_control=not scope.primary(m), prediction_error_veto_applied=False) for (c, m), v in grouped.items()]
    wave.csv_file(output/'metric_summary.csv', rows)
    with (original/'fit_quality.csv').open() as stream:
        wave.csv_file(output/'original_GP_fit_quality.csv', list(csv.DictReader(stream)))
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_subjects=data.n, original_GP_prediction_error_reproduction_maximum=maximum_fidelity_error))
    protocol.update(status='COMPLETE', scores_sha256=new_digest,
                    original_GP_prediction_error_reproduction_maximum=maximum_fidelity_error)
    wave.json_file(output/'protocol.json', protocol)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--source-run', default='waveAT_conditional_X_witness_GP_seed416')
    p.add_argument('--run-name', default='waveBA_GP_posterior_UX_seed416')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args(); source = wave.io.default_step2(wave.ROOT, args.model)
    parent = source.parent/'step5_matrix_metrics'
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        rows = analyze(source, parent/args.source_run, parent/args.run_name, device=args.device)
    print('[WaveBA] Primary grid candidates:', sum(r['all_planes_pass'] and not r['is_control'] for r in rows), flush=True)
