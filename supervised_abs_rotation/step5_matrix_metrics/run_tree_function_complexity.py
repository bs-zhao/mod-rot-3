"""Wave AU: direct-matrix conditional UX function tree description complexity."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import conditional_x_witness as witness
import cp_regression as cp
import tree_function_complexity as trees


def analyze(source, output, bank_folder, seed=421):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous tree complexity experiment: {output}')
    bank = profiles.MatrixBank(source, bank_folder); data = bank.data
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(Path(source).resolve()),
        seed=seed, channels=data.channels, source_matrix_manifest_sha256=bank.protocol['manifest_sha256'],
        original_response_reference=bank.quality['mode'], all_original_rows=data.m, status='RUNNING',
        X_contrast='W-only conditional feature witness, same protocol as AT',
        approximation='multioutput axis-aligned CART, nested training-risk best-first split path',
        relative_L2_budgets=[.01, .03, .1], output_mode_relative_squared_tolerance=1e-6,
        full_tree_heldout_relative_mse_maximum=.01, all_subjects_used_in_complexity_fit=True,
        sklearn_version=sklearn.__version__, primary_scope='conditional UX function rule-description complexity',
        score_sign='lower for every dataset', global_tree_search_optimality_verified=False,
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_tree_function_complexity.py', 'tree_function_complexity.py', 'conditional_x_witness.py',
             'matrix_function_profiles.py', 'matrix_io.py', 'cp_regression.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    maps, reveal, registration = [], {}, []
    for k, path in enumerate(data.paths):
        q, error = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); registration.append(error); reveal[f'case_{k:03d}'] = path.parent.name
    operators, audit = witness.operators(bank.x)
    wave.json_file(output/'X_crossfit_audit.json', audit)
    for j, operator in enumerate(operators):
        np.save(output/f'X_contrast_column{j}.npy', operator, allow_pickle=False)
    tasks, targets, discarded, audits = profiles.task_modes(bank.responses, operators)
    wave.json_file(output/'matrix_function_modes.json', dict(tasks=audits, response_quality=bank.quality))
    z = data.u-data.u.mean(0); z /= np.sqrt(np.var(data.u, axis=0).mean())
    classes, membership = cp.canonical_classes(z, maps)
    train = np.sort(np.random.default_rng(seed).permutation(data.n)[:int(.75*data.n)])
    valid = np.setdiff1d(np.arange(data.n), train); all_subjects = np.arange(data.n)
    np.savez_compressed(output/'geometry.npz', W_only_classes=classes, membership=membership, training=train, validation=valid)
    costs = np.zeros((len(classes), len(data.channels), 3, data.dx))
    worst_validation = np.zeros(len(data.channels)); fit_floor = worst_validation.copy(); task_rows = []
    for group, q in enumerate(classes):
        u = (z @ q).astype(np.float32)
        paths = {}
        for t, (c, j) in enumerate(tasks):
            leaves, errors, cv = trees.fit_path(u, targets[t], train, valid, discarded[t], seed=seed)
            paths[f'channel{c}_X{j}_CV_leaves'] = leaves
            paths[f'channel{c}_X{j}_CV_errors'] = errors
            leaves, errors, full = trees.fit_path(u, targets[t], all_subjects, all_subjects, discarded[t], seed=seed)
            paths[f'channel{c}_X{j}_all_subject_leaves'] = leaves
            paths[f'channel{c}_X{j}_all_subject_errors'] = errors
            costs[group, c, :, j] = [trees.minimum_leaves(leaves, errors[0], eps) for eps in (.01, .03, .1)]
            worst_validation[c] = max(worst_validation[c], cv['full_query_relative_mse'])
            fit_floor[c] = max(fit_floor[c], full['full_training_relative_mse'])
            task_rows.append(dict(channel=data.channels[c], X_column=j, coordinate_class=group,
                minimum_validation_relative_mse=cv['minimum_query_relative_mse'],
                full_validation_relative_mse=cv['full_query_relative_mse'],
                all_subject_full_tree_relative_mse=full['full_training_relative_mse'],
                all_subject_full_leaves=full['full_leaves'], all_subject_full_depth=full['full_depth']))
        np.savez_compressed(output/f'distortion_paths_class{group:03d}.npz', **paths)
        print(f'[WaveAU] {data.index["model"]}: all-subject tree paths {group+1}/{len(classes)}', flush=True)
    blind, quality = [], []
    for c, channel in enumerate(data.channels):
        quality.append(dict(estimator='matrix_conditional_tree', channel=channel,
            adequate=bank.quality['validation_nmse'][c] <= .1 and worst_validation[c] <= .01 and fit_floor[c] <= 1.1e-6,
            validation_nmse=bank.quality['validation_nmse'][c],
            worst_full_tree_subject_validation_relative_mse=worst_validation[c],
            worst_all_subject_full_tree_relative_mse=fit_floor[c]))
        for k, group in enumerate(membership):
            for metric, value in trees.scores(costs[group, c]).items():
                blind.append(dict(anonymous_case=f'case_{k:03d}', estimator='matrix_conditional_tree',
                                  channel=channel, metric=metric, loss=value))
    wave.csv_file(output/'blind_scores.csv', blind); wave.csv_file(output/'fit_quality.csv', quality)
    wave.csv_file(output/'tree_task_quality.csv', task_rows)
    np.savez_compressed(output/'per_X_complexity.npz', counts=costs)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict); qm = {r['channel']: r for r in quality}
    for row in blind:
        grouped[row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = [dict(estimator='matrix_conditional_tree', channel=c, metric=m,
        **wave.st.evaluate_curve(values, data.index['cases']), adequate=qm[c]['adequate'],
        validation_nmse=qm[c]['validation_nmse'], is_control=False) for (c, m), values in grouped.items()]
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_subjects_in_tree_fit=data.n, original_rows_in_response_preparation=data.m,
        source_matrix_manifest_sha256=bank.protocol['manifest_sha256'],
        original_U_model_coefficients_read=False, registration_max_residual=max(registration)))
    protocol.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', protocol)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3'); p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveAU_conditional_UX_tree_complexity_seed421')
    args = p.parse_args(); survivors = None
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            folder = parent/'matrix_response_table_seed383'
            if not folder.exists():
                profiles.prepare(source, folder)
            rows = analyze(source, parent/args.run_name, folder)
            passed = {(r['channel'], r['metric']) for r in rows if r['adequate'] and r['all_planes_pass']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveAU] {model}: {len(passed)} adequate tree-complexity passes, {len(survivors)} common', flush=True)
            if args.batch and not survivors:
                print('[WaveAU] No identical adequate criterion remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
