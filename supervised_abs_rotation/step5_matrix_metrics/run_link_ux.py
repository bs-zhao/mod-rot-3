"""Wave U: all-trial, scale-free UX geometry with output-link nuisance removed."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import joint_bank as jb
import link_ux as lu


def analyze(source, output, bank_folder, seed=400):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain old experiment: {output}')
    bank = jb.JointBank(source, bank_folder); data = bank.data
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(source.resolve()),
        seed=seed, bank_seed=bank.protocol['seed'], degrees=bank.protocol['degrees'], channels=list(data.channels),
        all_subjects_used=True, all_original_rows_used_in_fit=True, all_original_rows_used_in_metric=True,
        source_signature_sha256=bank.protocol['manifest_sha256'], adequacy_validation_nmse=.1,
        minimum_active_subject_fraction=.95, profiles=list(lu.PROFILES),
        score_sign='lower for all datasets', derivative_units='common standardized U, train-standardized X',
        descriptor='local UX geometry after optional scalar gradient-outer-product or X-tangent projection',
        pooling='local scale balancing, all-trial per-subject covariance, support, equal-subject aggregation',
        aggregation=['mean', 'median'], status='RUNNING',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_link_ux.py', 'link_ux.py', 'all_trial_moments.py', 'joint_fields.py', 'local_support.py',
             'fast_projection.py', 'joint_bank.py', 'matrix_io.py', 'surrogates.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', config)
    maps, reveal, residuals = [], {}, []
    for k, path in enumerate(data.paths):
        q, residual = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); residuals.append(residual); reveal[f'case_{k:03d}'] = path.parent.name
    blind, quality, hashes, audits = [], [], {}, {}
    for degree in bank.protocol['degrees']:
        name = f'poly{degree}'
        def progress(subjects, rows):
            print(f'[WaveU] {data.index["model"]} {name}: {subjects}/{data.n} people; {rows:,}/{data.m:,} actual rows', flush=True)
        moments, audit = lu.moments(data, bank.load(degree), progress=progress)
        path = output/f'ux_geometry_{name}.npz'; np.savez_compressed(path, **moments)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest(); audits[name] = audit
        for profile, covariance in moments.items():
            for c, channel in enumerate(data.channels):
                qr = bank.quality[name, channel]
                fraction = float(np.mean(np.trace(covariance[:, c], axis1=-2, axis2=-1).sum(-1) > 0))
                quality.append(dict(estimator=name, channel=channel, profile=profile,
                    adequate=float(qr['validation_nmse']) <= .1 and fraction >= .95,
                    validation_nmse=float(qr['validation_nmse']), active_subject_fraction=fraction))
                for k, q in enumerate(maps):
                    for metric, value in lu.scores(covariance[:, c], q).items():
                        key = 'control_'+profile+'_'+metric[8:] if metric.startswith('control_') else profile+'_'+metric
                        blind.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                                          metric=key, loss=value))
            print(f'[WaveU] {data.index["model"]} {name} {profile}: anonymous scores saved', flush=True)
        wave.csv_file(output/'blind_scores.csv', blind); wave.csv_file(output/'fit_quality.csv', quality)
        wave.json_file(output/'geometry_audit.json', audits)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    qm = {(r['estimator'], r['channel'], r['profile']): r for r in quality}; rows = []
    for (name, channel, metric), values in grouped.items():
        control = metric.startswith('control_')
        prefix = metric[8:] if control else metric
        profile = next(p for p in lu.PROFILES if prefix.startswith(p+'_'))
        qr = qm[name, channel, profile]
        if control and np.ptp(list(values.values())) > 1e-7*max(1., max(values.values())):
            raise ArithmeticError('UX geometry covariance norm changed under U rotation')
        rows.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index['cases']), adequate=qr['adequate'],
            validation_nmse=qr['validation_nmse'], is_control=control))
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_subjects=data.n, all_original_rows_in_fit_and_metric=data.m,
        geometry_moment_file_sha256=hashes, registration_max_residual=max(residuals)))
    config.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', config)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveU_link_ux_seed400')
    p.add_argument('--bank-name', default='joint_bank_seed391')
    args = p.parse_args(); survivors = None
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            rows = analyze(source, parent/args.run_name, parent/args.bank_name)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows
                      if r['adequate'] and r['all_planes_pass'] and not r['is_control']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveU] {model}: {len(passed)} passes; {len(survivors)} common candidates', flush=True)
            if args.batch and not survivors:
                print('[WaveU] No identical candidate remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
