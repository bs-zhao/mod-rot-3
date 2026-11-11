"""Wave V: all-trial UX support from a U-RBF times generic X dictionary."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import fitted_bank
import prepare_response_bank
import product_response as pr
import link_ux as lu


def analyze(source, output, bank_folder, seed=401):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain prior product-kernel experiment: {output}')
    bank = fitted_bank.FittedBank(source, bank_folder); data = bank.data
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(source.resolve()),
        seed=seed, source_seed=bank.protocol['seed'], estimators=bank.protocol['estimators'],
        channels=list(data.channels), profiles=list(pr.PROFILES), source_signature_sha256=bank.source_digest,
        source_code_sha256=bank.protocol['code_sha256'], all_subjects_used=True,
        all_original_rows_used_in_fit=True, all_original_rows_used_in_metric=True,
        X_dictionary='complete degree2 plus 64-center isotropic RBF length2; original-X training scales',
        X_ridge_grid=[1e-8, 1e-6, 1e-4, .01], adequacy_raw_validation_nmse=.1,
        adequacy_X_interpolation_nmse=.1, minimum_active_subject_fraction=.95,
        score_sign='lower for every dataset', status='RUNNING',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_product_response.py', 'product_response.py', 'fitted_bank.py', 'prepare_response_bank.py',
             'link_ux.py', 'all_trial_moments.py', 'joint_fields.py', 'local_support.py', 'fast_projection.py',
             'matrix_io.py', 'surrogates.py', 'structure.py', 'run_finite.py', 'additive_profiles.py'))).hexdigest())
    wave.json_file(output/'protocol.json', config)
    maps, reveal, residuals = [], {}, []
    for k, path in enumerate(data.paths):
        q, residual = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); residuals.append(residual); reveal[f'case_{k:03d}'] = path.parent.name
    dictionary = pr.XDictionary(data, seed=bank.protocol['seed'])
    with np.load(bank.folder/'response_bank.npz', allow_pickle=False) as saved:
        probes = saved['x']
    mapping, x_quality = pr.x_projection(bank, dictionary, probes, seed)
    wave.json_file(output/'X_interpolation_quality.json', x_quality)
    np.savez_compressed(output/'X_projection.npz', mapping=mapping, mean=dictionary.scaler.mean,
        scale=dictionary.scaler.scale, center_row_ids=dictionary.center_ids,
        normalized_centers=dictionary.kernel.centers, kernel_transform=dictionary.kernel.transform)
    blind, quality, hashes, audits = [], [], {}, {}
    for name in bank.protocol['estimators']:
        def progress(subjects, rows):
            print(f'[WaveV] {data.index["model"]} {name}: {subjects}/{data.n} people, {rows:,}/{data.m:,} raw rows', flush=True)
        moments, audit = pr.moments(bank, dictionary, name, mapping, progress)
        path = output/f'product_UX_moments_{name}.npz'; np.savez_compressed(path, **moments)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest(); audits[name] = audit
        for profile, covariance in moments.items():
            for c, channel in enumerate(data.channels):
                raw_nmse = audit['raw_heldout_subject_validation_nmse'][c]
                fraction = float(np.mean(np.trace(covariance[:, c], axis1=-2, axis2=-1).sum(-1) > 0))
                old = bank.quality[name, channel]
                adequate = (raw_nmse <= .1 and x_quality['validation_nmse'][c] <= .1 and
                            old['adequate'] == 'True' and fraction >= .95)
                quality.append(dict(estimator=name, channel=channel, profile=profile, adequate=adequate,
                    validation_nmse=raw_nmse, X_interpolation_nmse=x_quality['validation_nmse'][c],
                    original_U_bank_validation_nmse=float(old['validation_nmse']), active_subject_fraction=fraction))
                for k, q in enumerate(maps):
                    for metric, value in lu.scores(covariance[:, c], q).items():
                        key = 'control_'+profile+'_'+metric[8:] if metric.startswith('control_') else profile+'_'+metric
                        blind.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                                          metric=key, loss=value))
            print(f'[WaveV] {data.index["model"]} {name} {profile}: anonymous scores saved', flush=True)
        wave.csv_file(output/'blind_scores.csv', blind); wave.csv_file(output/'fit_quality.csv', quality)
        wave.json_file(output/'raw_prediction_audit.json', audits)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    qm = {(r['estimator'], r['channel'], r['profile']): r for r in quality}; rows = []
    for (name, channel, metric), values in grouped.items():
        control = metric.startswith('control_'); prefix = metric[8:] if control else metric
        profile = next(p for p in pr.PROFILES if prefix.startswith(p+'_'))
        qr = qm[name, channel, profile]
        if control and np.ptp(list(values.values())) > 1e-7*max(1., max(values.values())):
            raise ArithmeticError('Product-function UX norm changed under U rotation')
        rows.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index['cases']), adequate=qr['adequate'],
            validation_nmse=qr['validation_nmse'], is_control=control))
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)),
        derived_function_files=sorted(bank.derived_reads), forbidden_files_read=[],
        all_subjects=data.n, all_original_rows_in_fit_and_metric=data.m,
        product_moment_file_sha256=hashes, registration_max_residual=max(residuals)))
    config.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', config)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveV_product_kernel_seed401')
    p.add_argument('--bank-name', default='product_response_bank_seed383')
    args = p.parse_args(); survivors = None
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            bank_folder = parent/args.bank_name
            if not bank_folder.exists():
                prepare_response_bank.prepare(source, bank_folder, estimators=('rbf1', 'rbf2'))
            rows = analyze(source, parent/args.run_name, bank_folder)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows
                      if r['adequate'] and r['all_planes_pass'] and not r['is_control']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveV] {model}: {len(passed)} passes; {len(survivors)} common candidates', flush=True)
            if args.batch and not survivors:
                print('[WaveV] No identical candidate remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
