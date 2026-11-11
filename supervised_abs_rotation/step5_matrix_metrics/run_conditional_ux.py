"""Wave AE: empirical conditional-prediction UX interaction support."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz, save_npz, csr_matrix
from threadpoolctl import threadpool_limits

import run as wave
import fitted_bank
import prepare_response_bank
import ux_predictive_complexity as uc
import conditional_ux as conditional


def analyze(source, output, bank_folder, seed=407, estimators=None,
            capacities=((3, 64), (5, 128)), folds=5):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous conditional UX experiment: {output}')
    bank = fitted_bank.FittedBank(source, bank_folder); data = bank.data
    estimators = tuple(estimators or bank.protocol['estimators'])
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(source.resolve()),
        seed=seed, source_manifest_sha256=bank.source_digest, source_code_sha256=bank.protocol['code_sha256'],
        source_seed=bank.protocol['seed'], channels=list(data.channels), estimators=estimators,
        capacities=[list(c) for c in capacities], U_RBF_lengthscale=2., U_crossfit_folds=folds,
        U_ridge_grid=[1e-8, 1e-6, 1e-4, .01, .1], affine_complement_unpenalized=True,
        U_and_X_coordinates_independent=False, reference='empirical whole-U times observed joint-X blocks',
        definition='||R_U_without_i f_UX R_X_without_j^T||_F^2 / (subjects * X nodes)',
        primary_scope='UX sparsity; per-X normalized support primary, total strength is a control',
        all_subjects_used_in_function_fit_and_crossfit_scoring=True, all_original_rows_in_source_fit=True,
        score_sign='lower for every dataset', status='RUNNING',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_conditional_ux.py', 'conditional_ux.py', 'ux_predictive_complexity.py',
             'tail_support.py', 'fitted_bank.py', 'matrix_io.py', 'surrogates.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', config)
    maps, reveal, registration_errors = [], {}, []
    for k, path in enumerate(data.paths):
        q, error = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); registration_errors.append(error); reveal[f'case_{k:03d}'] = path.parent.name
    directions, memberships = conditional.direction_classes(maps)
    np.savez_compressed(output/'W_only_directions.npz', directions=directions, memberships=memberships)
    with np.load(bank.folder/'response_bank.npz', allow_pickle=False) as saved:
        x = saved['x']
    scale = wave.sur.Scaler(data.w[:, data.d:], 1).scale
    # Reuse the exact, frozen W/Y-derived X contrasts from Z when available.
    cached = output.parent/'waveZ_UX_complexity_seed405'
    if cached.exists():
        old = json.loads((cached/'protocol.json').read_text())
        digest = hashlib.sha256((cached/'blind_scores.csv').read_bytes()).hexdigest()
        if old['status'] != 'COMPLETE' or old['scores_sha256'] != digest or old['source_signature_sha256'] != bank.source_digest:
            raise ValueError('Frozen X-omission cache does not match the fitted function bank')
        operators = {f'omit_d{degree}': [load_npz(cached/f'X_omit_d{degree}_column{j}.npz').toarray()
                     for j in range(data.dx)] for degree in (2, 4)}
        x_audit = dict(source_run=str(cached.resolve()), frozen_scores_sha256=digest,
            maps_sha256={f'{mode}_X{j}': hashlib.sha256((cached/f'X_{mode}_column{j}.npz').read_bytes()).hexdigest()
                         for mode, matrices in operators.items() for j in range(len(matrices))})
    else:
        operators, x_audit = uc.omission_operators(x, scale, bank.bank, bank.train, seed=405)
    for mode, matrices in operators.items():
        for j, residual in enumerate(matrices):
            save_npz(output/f'X_{mode}_column{j}.npz', csr_matrix(residual))
    wave.json_file(output/'X_crossfit_audit.json', x_audit)
    blind, quality = [], []
    for name in estimators:
        coefficients, z, meta = uc.source_coefficients(bank, name)
        phi = conditional.source_features(z, meta)
        latents, weights, grams, compression = conditional.balanced_latents(phi, coefficients, operators)
        wave.json_file(output/f'{name}_latent_audit.json', compression)
        print(f'[WaveAE] {data.index["model"]} {name}: {len(directions)} directions, '
              f'{compression["latent_rank"]} latent functions, worst task reconstruction error '
              f'{compression["maximum_task_relative_squared_reconstruction_error"]:.3g}', flush=True)
        for degree, centers in capacities:
            capacity = f'poly{degree}_rbf{centers}'
            covariances, history = [], []
            for i, direction in enumerate(directions):
                covariance, audit = conditional.omission_covariance(z, direction, latents, weights,
                    degree=degree, centers=centers, seed=seed, folds=folds)
                covariances.append(covariance); history.append(dict(direction_id=i, **audit))
                if i % 8 == 0 or i+1 == len(directions):
                    print(f'[WaveAE] {name} {capacity}: conditional directions {i+1}/{len(directions)}', flush=True)
                    np.savez_compressed(output/f'{name}_{capacity}_covariance.npz', covariance=np.asarray(covariances))
                    wave.json_file(output/f'{name}_{capacity}_crossfit_audit.json', history)
            for k, membership in enumerate(memberships):
                for mode, gram in grams.items():
                    energy = conditional.energies(covariances, membership, gram)
                    for c, channel in enumerate(data.channels):
                        for metric, loss in conditional.sparsity(energy[c]).items():
                            blind.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                                metric=capacity+'_'+mode+'_'+metric, loss=loss))
            for c, channel in enumerate(data.channels):
                old = bank.quality[name, channel]
                quality.append(dict(estimator=name, channel=channel, capacity=capacity,
                    adequate=old['adequate'] == 'True' and compression['maximum_task_relative_squared_reconstruction_error'] <= 1.1e-7,
                    validation_nmse=float(old['validation_nmse']),
                    latent_rank=latents.shape[1], crossfit_subjects=data.n))
            wave.csv_file(output/'blind_scores.csv', blind); wave.csv_file(output/'fit_quality.csv', quality)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    qm = {(r['estimator'], r['channel'], r['capacity']): r for r in quality}; rows = []
    for (name, channel, metric), values in grouped.items():
        capacity = '_'.join(metric.split('_')[:2]); qr = qm[name, channel, capacity]
        rows.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index['cases']), adequate=qr['adequate'],
            validation_nmse=qr['validation_nmse'], is_control=False))
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)),
        derived_function_files=sorted(bank.derived_reads), forbidden_files_read=[],
        all_subjects_in_fit_and_crossfit_scoring=data.n, original_rows_in_source_fit=data.m,
        scored_X_nodes=len(x), registration_max_residual=max(registration_errors)))
    config.update(status='COMPLETE', scores_sha256=digest, W_only_unique_directions=len(directions))
    wave.json_file(output/'protocol.json', config)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveAE_conditional_UX_seed407')
    p.add_argument('--bank-name', default='UX_complexity_source_seed383')
    args = p.parse_args(); survivors = None
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            folder = parent/args.bank_name
            if not folder.exists():
                prepare_response_bank.prepare(source, folder, estimators=('poly3', 'poly5', 'rbf1', 'rbf2'))
            rows = analyze(source, parent/args.run_name, folder)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows
                      if r['adequate'] and r['all_planes_pass']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveAE] {model}: {len(passed)} UX passes; {len(survivors)} common criteria', flush=True)
            if args.batch and not survivors:
                print('[WaveAE] No identical criterion remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
