"""Wave Q: average effective support of local effects, not pooled energy."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

import run as wave
import run_finite
import local_support as ls


def analyze(source, fields, output, seed=396):
    source, fields, output = Path(source), Path(fields), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain existing result: {output}')
    old = json.loads((fields/'protocol.json').read_text())
    if old['status'] != 'COMPLETE' or hashlib.sha256((fields/'blind_scores.csv').read_bytes()).hexdigest() != old['scores_sha256']:
        raise ValueError('Incomplete or modified source function experiment')
    data = wave.io.MatrixCases(source, tuple(old['channels']), old['seed'])
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(source.resolve()),
        seed=seed, source_run_name=fields.name, source_code_sha256=old['code_sha256'],
        estimators=old['estimators'], channels=list(data.channels), all_subjects_used=True,
        pooling=['point', 'subject', 'context'], aggregation=['mean', 'median'],
        score_sign='lower for all datasets', adequacy_validation_nmse=.1, status='RUNNING',
        contrast='one nearest observed-X neighbor per stimulus; not an X partial derivative',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_local_support.py', 'local_support.py', 'matrix_io.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', config)
    maps, reveal, residuals = [], {}, []
    rng = np.random.default_rng(seed)
    for k, index in enumerate(rng.permutation(len(data.paths))):
        path = data.paths[index]
        q, residual = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); residuals.append(residual); reveal[f'case_{k:03d}'] = path.parent.name
    with np.load(fields/'response_bank.npz', allow_pickle=False) as saved:
        x_all = saved['x']
    with (fields/'fit_quality.csv').open(newline='') as stream:
        quality = {(r['estimator'], r['channel']): r for r in csv.DictReader(stream)}
    blind, source_files = [], {}
    for name in old['estimators']:
        with np.load(fields/f'model_{name}.npz', allow_pickle=False) as saved:
            x = x_all[saved['score_stimuli']]
        field_path = fields/f'field_{name}.npz'
        source_files[field_path.name] = hashlib.sha256(field_path.read_bytes()).hexdigest()
        with np.load(field_path, allow_pickle=False) as saved:
            g, h = saved['gradient'], saved['hessian']
            if len(np.unique(saved['subjects'])) != data.n:
                raise ValueError('This protocol requires every subject in the common derivative support')
        scale = np.std(x, axis=0); scale[scale < 1e-10] = 1.
        neighbor = cKDTree(x/scale).query(x/scale, k=2)[1][:, 1]
        edges = np.asarray(sorted({tuple(sorted((i, int(j)))) for i, j in enumerate(neighbor) if i != j}))
        for c, channel in enumerate(data.channels):
            delta = g[:, c, edges[:, 0]]-g[:, c, edges[:, 1]]
            for k, q in enumerate(maps):
                for metric, value in ls.evaluate(g[:, c], h[:, c], delta, q).items():
                    blind.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                                      metric=metric, loss=value))
                if k % 12 == 0:
                    print(f'[WaveQ] {data.index["model"]} {name} {channel} {k+1}/{len(maps)}', flush=True)
        wave.csv_file(output/'blind_scores.csv', blind)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = []
    for (name, channel, metric), values in grouped.items():
        control = metric.startswith('control_'); qr = quality[name, channel]
        if control and np.ptp(list(values.values())) > 1e-7*max(1., max(values.values())):
            raise ArithmeticError('Local invariant field norm changed')
        rows.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index['cases']), adequate=qr['adequate'] == 'True',
            validation_nmse=qr['validation_nmse'], is_control=control))
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.csv_file(output/'fit_quality.csv', list(quality.values()))
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_subjects=data.n, all_trial_rows=data.m, registration_max_residual=max(residuals),
        source_scores_sha256=old['scores_sha256'], source_field_sha256=source_files))
    config.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', config)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveQ_local_support_seed396')
    p.add_argument('--source-run', default='waveC_finite_seed383')
    args = p.parse_args(); survivors = None
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            fields = parent/args.source_run
            if not (fields/'protocol.json').exists():
                run_finite.analyze(source, fields)
            rows = analyze(source, fields, parent/args.run_name)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows
                      if r['adequate'] and r['all_planes_pass'] and not r['is_control']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveQ] {model}: {len(passed)} passes; {len(survivors)} common candidates', flush=True)
            if args.batch and not survivors:
                print('[WaveQ] No candidate remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
