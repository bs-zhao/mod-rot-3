"""Wave T: exact empirical derivative support using every recorded trial."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import joint_bank as jb
import all_trial_moments as am


def analyze(source, output, bank_folder, seed=399):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain existing all-trial experiment: {output}')
    bank = jb.JointBank(source, bank_folder); data = bank.data
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(source.resolve()),
        seed=seed, bank_seed=bank.protocol['seed'], degrees=bank.protocol['degrees'], channels=list(data.channels),
        all_subjects_used=True, all_original_rows_used_in_fit=True, all_original_rows_used_in_metric=True,
        source_signature_sha256=bank.protocol['manifest_sha256'], adequacy_validation_nmse=.1,
        score_sign='lower for all datasets', derivative_units='common standardized U, train-standardized X',
        pooling='exact all-trial energy within subject, then support, then equal-subject aggregation',
        joint_support='sum UU-edge and UX-entry support within each subject, then aggregate',
        aggregation=['mean', 'median'], status='RUNNING',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_all_trial_support.py', 'all_trial_moments.py', 'joint_fields.py', 'local_support.py',
             'fast_projection.py', 'joint_bank.py', 'matrix_io.py', 'surrogates.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', config)
    maps, reveal, residuals = [], {}, []
    for k, path in enumerate(data.paths):
        q, residual = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); residuals.append(residual); reveal[f'case_{k:03d}'] = path.parent.name
    blind, quality, moment_hashes = [], [], {}
    for degree in bank.protocol['degrees']:
        name = f'poly{degree}'; saved = bank.load(degree)
        def progress(subjects, rows):
            print(f'[WaveT] {data.index["model"]} {name}: exact moments {subjects}/{data.n} people, {rows:,}/{data.m:,} rows', flush=True)
        moments = am.moments(data, saved, progress=progress)
        moment_path = output/f'all_trial_moments_{name}.npz'
        np.savez_compressed(moment_path, **moments)
        moment_hashes[moment_path.name] = hashlib.sha256(moment_path.read_bytes()).hexdigest()
        for c, channel in enumerate(data.channels):
            qr = bank.quality[name, channel]
            quality.append(dict(estimator=name, channel=channel, adequate=float(qr['validation_nmse']) <= .1,
                                validation_nmse=float(qr['validation_nmse'])))
            for k, q in enumerate(maps):
                scores = am.scores(moments['UU'][:, c], moments['UX'][:, c], moments['gradient'][:, c], q)
                for metric, value in scores.items():
                    blind.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                                      metric=metric, loss=value))
            print(f'[WaveT] {data.index["model"]} {name} {channel}: all anonymous cases saved', flush=True)
        wave.csv_file(output/'blind_scores.csv', blind)
        wave.csv_file(output/'fit_quality.csv', quality)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    qm = {(r['estimator'], r['channel']): r for r in quality}; rows = []
    for (name, channel, metric), values in grouped.items():
        control = metric.startswith('control_')
        if control and np.ptp(list(values.values())) > 1e-7*max(1., max(values.values())):
            raise ArithmeticError('All-trial derivative covariance control changed')
        rows.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index['cases']), adequate=qm[name, channel]['adequate'],
            validation_nmse=qm[name, channel]['validation_nmse'], is_control=control))
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_subjects=data.n, all_original_rows_in_fit_and_metric=data.m, moment_file_sha256=moment_hashes,
        registration_max_residual=max(residuals)))
    config.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', config)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveT_all_trials_seed399')
    p.add_argument('--bank-name', default='joint_bank_seed391')
    args = p.parse_args(); survivors = None; seen = {}
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            rows = analyze(source, parent/args.run_name, parent/args.bank_name)
            seen[model] = rows
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows
                      if r['adequate'] and r['all_planes_pass'] and not r['is_control']}
            survivors = passed if survivors is None else survivors & passed
            wave.csv_file(wave.ROOT/'outputs'/'supervised_abs_rotation'/f'step5_{args.run_name}_screen.csv',
                [dict(model=m, **r) for m, result in seen.items() for r in result])
            print(f'[WaveT] {model}: {len(passed)} passes; {len(survivors)} common candidates', flush=True)
            if args.batch and not survivors:
                print('[WaveT] No identical candidate remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
