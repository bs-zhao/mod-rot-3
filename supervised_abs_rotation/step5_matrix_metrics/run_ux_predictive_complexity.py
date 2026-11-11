"""Wave Z: polynomial/GP-mean approximation, then UX interaction sparsity."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import save_npz, csr_matrix
from threadpoolctl import threadpool_limits

import run as wave
import fitted_bank
import prepare_response_bank
import ux_predictive_complexity as uc


def analyze(source, output, bank_folder, seed=405, anchors_count=8, sigmas=(.2, .5)):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous UX-complexity experiment: {output}')
    bank = fitted_bank.FittedBank(source, bank_folder); data = bank.data
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(source.resolve()),
        seed=seed, source_seed=bank.protocol['seed'], estimators=bank.protocol['estimators'],
        channels=list(data.channels), source_signature_sha256=bank.source_digest,
        source_code_sha256=bank.protocol['code_sha256'], all_subjects_used=True,
        all_original_rows_used_in_source_fit=True, all_source_X_nodes_scored=True,
        primary_scope='UX predictive interaction sparsity; first/total Ui effects of heldout-X omission contrasts',
        X_omission_degrees=[2, 4], X_omission_RBF_centers=32, X_omission_RBF_lengthscale=2.,
        X_crossfit_folds=5, X_inner_train_fraction=.75, X_ridge_grid=[1e-8, 1e-6, 1e-4, .01],
        X_affine_complement_unpenalized=True, X_Y_tuning_uses_training_U_subjects_only=True,
        empirical_input_independence_assumed=False, U_reference='within-anchor local isotropic Gaussian',
        anchors_count=anchors_count, sigmas=list(sigmas), minimum_reference_coverage=.9,
        support_k=16, support_factor=1.5, negative_relative_tolerance=1e-5,
        invariant_absolute_relative_tolerance=1e-6, score_sign='lower for every dataset', status='RUNNING',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_ux_predictive_complexity.py', 'ux_predictive_complexity.py', 'fitted_bank.py',
             'prepare_response_bank.py', 'kernel_anova.py', 'tail_support.py', 'matrix_io.py',
             'surrogates.py', 'structure.py', 'run_finite.py', 'additive_profiles.py'))).hexdigest())
    wave.json_file(output/'protocol.json', config)
    maps, reveal, residuals = [], {}, []
    for k, path in enumerate(data.paths):
        q, residual = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); residuals.append(residual); reveal[f'case_{k:03d}'] = path.parent.name
    with np.load(bank.folder/'response_bank.npz', allow_pickle=False) as saved:
        x = saved['x']
    scale = wave.sur.Scaler(data.w[:, data.d:], 1).scale
    operators, x_audit = uc.omission_operators(x, scale, bank.bank, bank.train, seed)
    for mode, matrices in operators.items():
        for j, residual in enumerate(matrices):
            save_npz(output/f'X_{mode}_column{j}.npz', csr_matrix(residual))
    wave.json_file(output/'X_crossfit_audit.json', x_audit)
    rng = np.random.default_rng(seed)
    anchor_ids = rng.choice(data.n, min(anchors_count, data.n), replace=False)
    np.save(output/'anchor_subject_ids.npy', anchor_ids)
    blind, quality, audits, numerics = [], [], [], defaultdict(list)
    for name in bank.protocol['estimators']:
        coefficients, z, meta = uc.source_coefficients(bank, name)
        grams = uc.coefficient_grams(coefficients, operators)
        anchors = z[anchor_ids]; tree = cKDTree(z)
        radius = tree.query(anchors, k=min(16, data.n))[0][:, -1]
        coverages = {}
        for sigma in sigmas:
            noise = np.random.default_rng(seed+1).normal(size=(len(anchors), 128, data.d))
            probes = (anchors[:, None]+sigma*noise).reshape(-1, data.d)
            shifted = tree.query(probes, k=min(16, data.n))[0][:, -1].reshape(len(anchors), -1)
            coverages[sigma] = float(np.mean(shifted <= 1.5*radius[:, None]))
        for k, q in enumerate(maps):
            for sigma in sigmas:
                tables = uc.moment_tables(meta, anchors, sigma, q)
                for mode, by_channel in grams.items():
                    for c, channel in enumerate(data.channels):
                        values, numeric = uc.energies_from_tables(tables, by_channel[c])
                        numerics[name, channel, sigma, mode].append(numeric['negative_relative_energy'])
                        prefix = f's{sigma:g}_{mode}_'
                        for order, energy in values.items():
                            for metric, loss in uc.sparsity(energy).items():
                                blind.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                                    metric=prefix+order+'_'+metric, loss=loss))
                        control = float(np.mean(numeric['total_U_variance_by_anchor_X']))
                        blind.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                            metric='control_'+prefix+'UX_residual_U_variance', loss=control))
                        audits.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                            sigma=sigma, X_mode=mode, U_first_energy=values['U_first'].tolist(),
                            U_total_energy=values['U_total'].tolist(), **numeric))
            if k % 12 == 0:
                print(f'[WaveZ] {data.index["model"]} {name} UX complexity {k+1}/{len(maps)}', flush=True)
        for channel in data.channels:
            old = bank.quality[name, channel]
            for sigma in sigmas:
                for mode in grams:
                    maximum_negative = max(numerics[name, channel, sigma, mode])
                    quality.append(dict(estimator=name, channel=channel, sigma=sigma, X_mode=mode,
                        adequate=old['adequate'] == 'True' and coverages[sigma] >= .9 and maximum_negative <= 1e-5,
                        validation_nmse=float(old['validation_nmse']), reference_coverage=coverages[sigma],
                        maximum_negative_relative_energy=maximum_negative))
        wave.csv_file(output/'blind_scores.csv', blind); wave.csv_file(output/'fit_quality.csv', quality)
        wave.json_file(output/'UX_energy_audit.json', audits)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    qm = {(r['estimator'], r['channel'], r['sigma'], r['X_mode']): r for r in quality}; rows = []
    for (name, channel, metric), values in grouped.items():
        control = metric.startswith('control_'); prefix = metric[8:] if control else metric
        sigma = float(prefix.split('_', 1)[0][1:])
        mode = next(m for m in operators if prefix.startswith(f's{sigma:g}_{m}_'))
        qr = qm[name, channel, sigma, mode]
        if control and np.ptp(list(values.values())) > 1e-6*max(1., max(map(abs, values.values()))):
            raise ArithmeticError('UX residual variance changed under U-reference rotation')
        rows.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index['cases']), adequate=qr['adequate'],
            validation_nmse=qr['validation_nmse'], is_control=control))
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)),
        derived_function_files=sorted(bank.derived_reads), forbidden_files_read=[], all_subjects=data.n,
        all_original_rows_in_source_fit=data.m, scored_X_nodes=len(x), registration_max_residual=max(residuals)))
    config.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', config)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveZ_UX_complexity_seed405')
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
                      if r['adequate'] and r['all_planes_pass'] and not r['is_control']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveZ] {model}: {len(passed)} UX sparsity passes; {len(survivors)} common criteria', flush=True)
            if args.batch and not survivors:
                print('[WaveZ] No identical UX criterion remains; later models skipped', flush=True)
                return


if __name__ == '__main__':
    main()
