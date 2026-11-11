"""Wave X: support of UX contrasts estimated on recorded X neighborhoods."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import save_npz
from threadpoolctl import threadpool_limits

import run as wave
import fitted_bank
import prepare_response_bank
import observed_x_contrasts as ox
import link_ux as lu


def analyze(source, output, bank_folder, seed=403, neighbors=(16, 32, 64)):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous contrast experiment: {output}')
    bank = fitted_bank.FittedBank(source, bank_folder); data = bank.data
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(source.resolve()),
        seed=seed, source_seed=bank.protocol['seed'], estimators=bank.protocol['estimators'], channels=list(data.channels),
        source_signature_sha256=bank.source_digest, source_code_sha256=bank.protocol['code_sha256'],
        all_subjects_used=True, all_original_rows_used_in_source_fit=True, all_source_X_nodes_scored=True,
        neighbors=list(neighbors), relative_singular_threshold=.001, minimum_full_rank_node_fraction=.95,
        profiles=['global', 'point', 'column'], minimum_active_subject_fraction=.95,
        adequacy_source_validation_nmse=.1, score_sign='lower for every dataset',
        descriptor='observed-X local linear contrasts of fitted U gradients; not unique pointwise X partials',
        status='RUNNING', code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_observed_x_contrasts.py', 'observed_x_contrasts.py', 'fitted_bank.py', 'prepare_response_bank.py',
             'link_ux.py', 'all_trial_moments.py', 'joint_fields.py', 'local_support.py', 'fast_projection.py',
             'matrix_io.py', 'surrogates.py', 'structure.py', 'run_finite.py', 'additive_profiles.py'))).hexdigest())
    wave.json_file(output/'protocol.json', config)
    maps, reveal, residuals = [], {}, []
    for k, path in enumerate(data.paths):
        q, residual = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); residuals.append(residual); reveal[f'case_{k:03d}'] = path.parent.name
    with np.load(bank.folder/'response_bank.npz', allow_pickle=False) as saved:
        x = saved['x']
    scale = wave.sur.Scaler(data.w[:, data.d:], 1).scale
    operators, geometry = {}, {}
    for k in neighbors:
        operators[k], geometry[k] = ox.contrast_operator(x, scale, k)
        save_npz(output/f'X_contrast_operator_k{k}.npz', operators[k])
    wave.json_file(output/'X_geometry_audit.json', geometry)
    blind, quality, hashes = [], [], {}
    for name in bank.protocol['estimators']:
        gradient = ox.source_gradient(bank, name)
        print(f'[WaveX] {data.index["model"]} {name}: U gradients at all {len(x)} source X nodes, {data.n} subjects', flush=True)
        for k, operator in operators.items():
            moments = ox.moments(gradient, operator, data.dx)
            path = output/f'contrast_moments_{name}_k{k}.npz'; np.savez_compressed(path, **moments)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            for profile, covariance in moments.items():
                for c, channel in enumerate(data.channels):
                    old = bank.quality[name, channel]
                    fraction = float(np.mean(np.trace(covariance[:, c], axis1=-2, axis2=-1).sum(-1) > 0))
                    adequate = (old['adequate'] == 'True' and fraction >= .95 and
                                geometry[k]['full_rank_node_fraction'] >= .95)
                    quality.append(dict(estimator=name, channel=channel, neighbor_count=k, profile=profile,
                        adequate=adequate, validation_nmse=float(old['validation_nmse']),
                        active_subject_fraction=fraction, full_rank_node_fraction=geometry[k]['full_rank_node_fraction']))
                    for i, q in enumerate(maps):
                        for metric, value in lu.scores(covariance[:, c], q).items():
                            prefix = f'k{k}_{profile}_'
                            key = 'control_'+prefix+metric[8:] if metric.startswith('control_') else prefix+metric
                            blind.append(dict(anonymous_case=f'case_{i:03d}', estimator=name,
                                              channel=channel, metric=key, loss=value))
            print(f'[WaveX] {data.index["model"]} {name} k{k}: all anonymous contrast scores saved', flush=True)
            wave.csv_file(output/'blind_scores.csv', blind); wave.csv_file(output/'fit_quality.csv', quality)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    qm = {(r['estimator'], r['channel'], r['neighbor_count'], r['profile']): r for r in quality}; rows = []
    for (name, channel, metric), values in grouped.items():
        control = metric.startswith('control_'); prefix = metric[8:] if control else metric
        k_text, profile = prefix.split('_', 2)[:2]; k = int(k_text[1:])
        qr = qm[name, channel, k, profile]
        if control and np.ptp(list(values.values())) > 1e-7*max(1., max(values.values())):
            raise ArithmeticError('Observed-X contrast norm changed under U rotation')
        rows.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index['cases']), adequate=qr['adequate'],
            validation_nmse=qr['validation_nmse'], is_control=control))
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)),
        derived_function_files=sorted(bank.derived_reads), forbidden_files_read=[],
        all_subjects=data.n, all_original_rows_in_source_fit=data.m, scored_X_nodes=len(x),
        contrast_moment_file_sha256=hashes, registration_max_residual=max(residuals)))
    config.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', config)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveX_observed_contrasts_seed403')
    p.add_argument('--bank-name', default='product_response_bank_seed383')
    args = p.parse_args(); survivors = None
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            folder = parent/args.bank_name
            if not folder.exists():
                prepare_response_bank.prepare(source, folder, estimators=('rbf1', 'rbf2'))
            rows = analyze(source, parent/args.run_name, folder)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows
                      if r['adequate'] and r['all_planes_pass'] and not r['is_control']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveX] {model}: {len(passed)} passes; {len(survivors)} common candidates', flush=True)
            if args.batch and not survivors:
                print('[WaveX] No identical candidate remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
