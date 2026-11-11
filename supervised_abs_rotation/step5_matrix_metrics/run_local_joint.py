"""Wave R: actual joint-W UU/UX derivatives with local effective support."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import joint_bank as jb
import joint_fields as jf
import local_joint as lj


def analyze(source, output, bank_folder, seed=397, per_subject=8):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain existing result: {output}')
    bank = jb.JointBank(source, bank_folder); data = bank.data
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(source.resolve()),
        seed=seed, bank_seed=bank.protocol['seed'], degrees=bank.protocol['degrees'], channels=list(data.channels),
        evaluation_rows_per_subject=per_subject, all_subjects_used=True, all_original_rows_used_in_fit=True,
        source_signature_sha256=bank.protocol['manifest_sha256'], adequacy_validation_nmse=.1,
        score_sign='lower for all datasets', derivative_units='common standardized U, train-standardized X',
        pooling=['point', 'subject', 'context'], aggregation=['mean', 'median'], status='RUNNING',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_local_joint.py', 'joint_fields.py', 'local_joint.py', 'local_support.py',
             'joint_bank.py', 'matrix_io.py', 'surrogates.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', config)
    maps, reveal, residuals = [], {}, []
    for k, path in enumerate(data.paths):
        q, residual = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); residuals.append(residual); reveal[f'case_{k:03d}'] = path.parent.name
    row_ids = jf.observed_rows(data, per_subject)
    np.save(output/'evaluation_row_ids.npy', row_ids)
    blind, quality = [], []
    for degree in bank.protocol['degrees']:
        name = f'poly{degree}'; saved = bank.load(degree)
        z = (np.asarray(data.w[row_ids], dtype=float)-saved['mean'])/saved['scale']
        g, h = jf.evaluate(z, saved['powers'], saved['weights'], saved['coef'])
        g = g.reshape(data.n, per_subject, len(data.channels), -1)
        h = h.reshape(data.n, per_subject, len(data.channels), z.shape[1], z.shape[1])
        np.savez_compressed(output/f'field_{name}.npz', gradient=g, hessian=h)
        for c, channel in enumerate(data.channels):
            qr = bank.quality[name, channel]
            quality.append(dict(estimator=name, channel=channel, adequate=float(qr['validation_nmse']) <= .1,
                                validation_nmse=float(qr['validation_nmse'])))
            for k, q in enumerate(maps):
                scores = lj.evaluate(g[:, :, c], h[:, :, c], q, data.d)
                for metric, value in scores.items():
                    blind.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                                      metric=metric, loss=value))
                if k % 12 == 0:
                    print(f'[WaveR] {data.index["model"]} {name} {channel} {k+1}/{len(maps)}', flush=True)
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
            raise ArithmeticError('Joint derivative control changed under orthogonal U rotation')
        rows.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index['cases']), adequate=qm[name, channel]['adequate'],
            validation_nmse=qm[name, channel]['validation_nmse'], is_control=control))
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_subjects=data.n, all_trial_rows_in_fit=data.m, evaluated_rows=len(row_ids),
        registration_max_residual=max(residuals)))
    config.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', config)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveR_local_joint_seed397')
    p.add_argument('--bank-name', default='joint_bank_seed391')
    args = p.parse_args(); survivors = None; seen = {}
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            folder = parent/args.bank_name
            if not folder.exists():
                jb.prepare(source, folder)
            rows = analyze(source, parent/args.run_name, folder)
            seen[model] = rows
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows
                      if r['adequate'] and r['all_planes_pass'] and not r['is_control']}
            survivors = passed if survivors is None else survivors & passed
            wave.csv_file(wave.ROOT/'outputs'/'supervised_abs_rotation'/f'step5_{args.run_name}_screen.csv',
                [dict(model=m, **r) for m, result in seen.items() for r in result])
            print(f'[WaveR] {model}: {len(passed)} passes; {len(survivors)} common candidates', flush=True)
            if args.batch and not survivors:
                print('[WaveR] No candidate remains; later datasets skipped', flush=True)
                return
        if args.batch:
            print(f'[WaveR] {len(survivors)} four-model grid candidates; stability and optimization still required', flush=True)


if __name__ == '__main__':
    main()
