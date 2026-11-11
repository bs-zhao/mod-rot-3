"""Wave AK: all-trial local UX support, normalized before pooling."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import joint_bank
import all_point_ux as points


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def analyze(source, output, folders, degrees=(4, 5), seed=410):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous all-point UX experiment: {output}')
    banks = [joint_bank.JointBank(source, folder) for folder in folders]
    data = banks[0].data
    if len(banks) != len(degrees) or any(not np.array_equal(bank.data.u, data.u) for bank in banks):
        raise ValueError('All fitted degrees must use the same physical W reference')
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(Path(source).resolve()),
        seed=seed, degrees=degrees, channels=data.channels, all_subjects_used=True,
        all_original_rows_in_fit_and_local_support=data.m,
        primary_scope='pointwise UX-column support before trial aggregation; all observed W rows',
        trial_pools=points.TRIAL_POOLS, X_pools=points.X_POOLS, subject_pools=['mean', 'median'],
        source_manifests={f'poly{degree}': bank.protocol['manifest_sha256'] for degree, bank in zip(degrees, banks)},
        score_sign='lower for every dataset', precision='float64', status='RUNNING',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_all_point_ux.py', 'all_point_ux.py', 'joint_fields.py', 'local_support.py',
             'joint_bank.py', 'fast_projection.py', 'matrix_io.py', 'surrogates.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', config)
    maps, reveal, registration_errors = [], {}, []
    for k, path in enumerate(data.paths):
        q, error = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); registration_errors.append(error); reveal[f'case_{k:03d}'] = path.parent.name
    blind, quality, fields = [], [], {}
    for degree, bank in zip(degrees, banks):
        name = f'poly{degree}'
        field = points.make_field(data, bank.load(degree), output/f'UX_field_{name}.npy')
        fields[name] = dict(file=f'UX_field_{name}.npy', sha256=file_digest(output/f'UX_field_{name}.npy'), rows=data.m)
        print(f'[WaveAK] {data.index["model"]} {name}: actual UX field at all {data.m} rows prepared', flush=True)
        for k, q in enumerate(maps):
            statistics, keys = points.subject_statistics(field, data.offsets, q)
            np.savez_compressed(output/f'{name}_subject_statistics_case{k:03d}.npz', values=statistics)
            for c, channel in enumerate(data.channels):
                for metric, loss in points.scores(statistics, keys, c).items():
                    blind.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel, metric=metric, loss=loss))
            wave.csv_file(output/'blind_scores.csv', blind)
            print(f'[WaveAK] {data.index["model"]} {name}: all-point case {k+1}/{len(maps)}', flush=True)
        for channel in data.channels:
            qr = bank.quality[name, channel]
            quality.append(dict(estimator=name, channel=channel, adequate=float(qr['validation_nmse']) <= .1,
                                validation_nmse=float(qr['validation_nmse'])))
        wave.csv_file(output/'fit_quality.csv', quality)
        wave.json_file(output/'subject_statistics_schema.json', dict(metric_keys=keys,
            axes=['subject', 'trial_pool', 'channel', 'X_pool', 'metric'],
            trial_pools=points.TRIAL_POOLS, X_pools=points.X_POOLS, channels=data.channels))
    digest = file_digest(output/'blind_scores.csv')
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict); qm = {(r['estimator'], r['channel']): r for r in quality}
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = [dict(estimator=e, channel=c, metric=m, **wave.st.evaluate_curve(values, data.index['cases']),
        adequate=qm[e, c]['adequate'], validation_nmse=qm[e, c]['validation_nmse'], is_control=False)
        for (e, c, m), values in grouped.items()]
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_subjects=data.n, all_original_rows_in_source_fit_and_point_scores=data.m,
        source_fields=fields, registration_max_residual=max(registration_errors)))
    config.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', config)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3'); p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveAK_all_point_UX_seed410')
    args = p.parse_args(); survivors = None
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            folders = [parent/'joint_bank_seed391', parent/'joint_bank_degree5_seed391']
            for degree, folder in zip((4, 5), folders):
                if not folder.exists():
                    joint_bank.prepare(source, folder, degrees=(degree,), device='cuda' if model == 'CR3' else 'auto')
            rows = analyze(source, parent/args.run_name, folders)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows if r['adequate'] and r['all_planes_pass']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveAK] {model}: {len(passed)} all-point UX passes, {len(survivors)} common', flush=True)
            if args.batch and not survivors:
                print('[WaveAK] No identical criterion remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
