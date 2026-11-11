"""Wave M: empirical-support prediction compression; CR3 first."""
import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import joint_bank as jb
import predictive_compression as pc


def analyze(source, output, bank_folder, seed=392, max_terms=128):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain existing experiment: {output}')
    bank = jb.JointBank(source, bank_folder); data = bank.data
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(source.resolve()),
        seed=seed, bank_seed=bank.protocol['seed'], degrees=bank.protocol['degrees'], channels=list(data.channels),
        max_terms=max_terms, adequacy_validation_nmse=.1, conditional_relative_energy_floor=1e-8,
        all_subjects_used=True, status='RUNNING', score_sign='lower for all datasets',
        target='fixed all-data joint projection; compression fit on 75% subjects, validation on 25%',
        source_signature_sha256=bank.protocol['manifest_sha256'],
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_predictive_compression.py', 'predictive_compression.py', 'coefficient_support.py',
             'kernel_anova.py', 'joint_bank.py', 'tail_support.py', 'matrix_io.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', config)
    maps, reveal, residuals = [], {}, []
    for k, path in enumerate(data.paths):
        q, residual = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); residuals.append(residual); reveal[f'case_{k:03d}'] = path.parent.name
    blind, quality, audits = [], [], []
    for degree in bank.protocol['degrees']:
        name = f'poly{degree}'; saved = bank.load(degree)
        transport = pc.Transport(saved['powers'], data.d, seed)
        system = pc.nuisance_system(saved['g_all']-saved['g_validation'], saved['g_validation'],
                                    saved['coef'], transport.structure.has_u)
        relative = system['ss_train']/np.maximum(system['full_train'], np.finfo(float).tiny)
        for c, channel in enumerate(data.channels):
            qr = bank.quality[name, channel]
            quality.append(dict(estimator=name, channel=channel, validation_nmse=float(qr['validation_nmse']),
                adequate=float(qr['validation_nmse']) <= .1 and relative[c] > 1e-8,
                conditional_relative_energy=float(relative[c]), x_nuisance_rank=system['x_rank']))
        for k, q in enumerate(maps):
            blocks = transport.blocks(q)
            cmat, vmat = (transport.square(system[key], blocks) for key in ('c', 'v'))
            ct, cv = (transport.vector(system[key], blocks) for key in ('ct', 'cv'))
            for c, channel in enumerate(data.channels):
                if relative[c] <= 1e-8:
                    raise ValueError('U prediction is numerically unidentifiable; no sparse recovery claim')
                scores, audit = pc.compress(cmat, vmat, ct[:, c], cv[:, c], system['ss_train'][c],
                                           system['ss_validation'][c], max_terms)
                for metric, loss in scores.items():
                    blind.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                                      metric=metric, loss=loss))
                audits.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel, **audit))
            if k % 12 == 0:
                print(f'[WaveM] {data.index["model"]} {name} anonymous {k+1}/{len(maps)}', flush=True)
        wave.csv_file(output/'blind_scores.csv', blind)
        wave.csv_file(output/'fit_quality.csv', quality)
        wave.json_file(output/'compression_audit.json', audits)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    quality_map = {(r['estimator'], r['channel']): r for r in quality}
    summaries = []
    for (name, channel, metric), values in grouped.items():
        qr = quality_map[name, channel]
        summaries.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index['cases']), adequate=qr['adequate'],
            validation_nmse=qr['validation_nmse'], is_control=False))
    wave.csv_file(output/'metric_summary.csv', summaries)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_subjects=data.n, all_trial_rows=data.m, registration_max_residual=max(residuals)))
    config.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', config)
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveM_prediction_seed392')
    p.add_argument('--bank-name', default='joint_bank_seed391')
    p.add_argument('--seed', type=int, default=392)
    p.add_argument('--max-terms', type=int, default=128)
    args = p.parse_args()
    if args.max_terms < 2:
        p.error('At least two terms required')
    survivors = None
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model)
            folder = source.parent/'step5_matrix_metrics'
            bank_folder, output = folder/args.bank_name, folder/args.run_name
            if not bank_folder.exists():
                jb.prepare(source, bank_folder)
            rows = analyze(source, output, bank_folder, args.seed, args.max_terms)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows
                      if r['adequate'] and r['all_planes_pass']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveM] {model}: {len(passed)} passes; {len(survivors)} common survivors', flush=True)
            if args.batch and not survivors:
                print('[WaveM] No common candidate remains; later models skipped', flush=True)
                return


if __name__ == '__main__':
    main()
