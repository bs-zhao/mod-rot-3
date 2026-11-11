"""Wave AO: conditional X-balanced, globally integrable link and UX support."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import joint_bank
import global_output_link as link
import equal_x_output_link as equal
from run_all_point_ux import file_digest


def analyze(fields, output, seed=412, bins=(0, 16, 64, 256), reference_bins=64):
    fields, output = Path(fields), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous balanced-link experiment: {output}')
    old = json.loads((fields/'protocol.json').read_text())
    digest = file_digest(fields/'blind_scores.csv')
    if old['status'] != 'COMPLETE' or old['scores_sha256'] != digest:
        raise ValueError('Source fields require a complete frozen AM experiment')
    source = Path(old['source']); parent = source.parent/'step5_matrix_metrics'
    folders = {4: parent/'joint_bank_seed391', 5: parent/'joint_bank_degree5_seed391'}
    banks = {degree: joint_bank.JointBank(source, folders[degree]) for degree in old['degrees']}
    for degree, bank in banks.items():
        if bank.protocol['manifest_sha256'] != old['source_manifests'][f'poly{degree}']:
            raise ValueError('Source derivative fields no longer match the immutable polynomial fit')
    data = next(iter(banks.values())).data
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(source.resolve()),
        seed=seed, source_fields=str(fields.resolve()), source_fields_scores_sha256=digest,
        degrees=old['degrees'], channels=data.channels, bins=bins, reference_bins=reference_bins,
        primary_scope='global output-link-corrected UX support with conditionally balanced X-column geometry',
        normalizer='E[||gU*gX_j||^2 | fitted response], shared reference64 for every candidate resolution',
        all_subjects_used_in_refit_and_moments=True, all_original_rows_in_fit_and_moments=data.m,
        selected_resolution_only_primary=True, score_sign='lower for every dataset', status='RUNNING',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_equal_x_output_link.py', 'equal_x_output_link.py', 'global_output_link.py', 'joint_bank.py',
             'matrix_io.py', 'local_support.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    maps, reveal, errors = [], {}, []
    for k, path in enumerate(data.paths):
        q, error = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); errors.append(error); reveal[f'case_{k:03d}'] = path.parent.name
    training = np.isin(data.sid, np.random.default_rng(seed).permutation(data.n)[:int(.75*data.n)])
    blind, quality, cache_hashes = [], [], {}
    for degree, bank in banks.items():
        name = f'poly{degree}'; folder = fields/name
        for filename in ('prediction.npy', 'gradient.npy', 'mixed.npy'):
            cache_hashes[f'{name}/{filename}'] = file_digest(folder/filename)
        y, g, h = [np.load(folder/f, mmap_mode='r', allow_pickle=False) for f in ('prediction.npy', 'gradient.npy', 'mixed.npy')]
        if y.shape != data.y.shape or g.shape != (data.m, len(data.channels), data.d+data.dx) or h.shape != (data.m, len(data.channels), data.dx, data.d):
            raise ValueError('Cached derivatives do not match actual case rows')
        numerator, denominator, base = equal.column_invariants(g, h, data.d)
        links, chosen, audits = equal.fit_all(y, numerator, denominator, base, training, bins, reference_bins)
        wave.json_file(output/f'{name}_global_links.json', dict(links=links, selected_bins=chosen, audits=audits))
        moments = link.moments(data, y, g, h, links, bins,
            progress=lambda count: print(f'[WaveAO] {data.index["model"]} {name}: moments {count}/{data.n}', flush=True))
        for count, matrix in moments.items():
            np.savez_compressed(output/f'{name}_bins{count}_moments.npz', UX=matrix)
        for c, channel in enumerate(data.channels):
            audit = next(r for r in audits[c]['history'] if r['bins'] == chosen[c])
            qr = bank.quality[name, channel]
            quality.append(dict(estimator=name, channel=channel,
                adequate=float(qr['validation_nmse']) <= .1 and audit['numerical_derivative_dynamic_range_representable'],
                validation_nmse=float(qr['validation_nmse']), selected_bins=chosen[c],
                log_derivative_range=audit['log_derivative_range'], global_link_integrability_verified=True))
            for k, q in enumerate(maps):
                for count, matrix in moments.items():
                    for metric, loss in link.scores(matrix[:, c], q).items():
                        blind.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                                          metric=f'control_bins{count}_'+metric, loss=loss))
                        if count == chosen[c]:
                            blind.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                                              metric='selected_equal_X_global_link_'+metric, loss=loss))
        wave.csv_file(output/'blind_scores.csv', blind); wave.csv_file(output/'fit_quality.csv', quality)
    digest = file_digest(output/'blind_scores.csv')
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict); qm = {(r['estimator'], r['channel']): r for r in quality}
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = [dict(estimator=e, channel=c, metric=m, **wave.st.evaluate_curve(v, data.index['cases']),
        adequate=qm[e, c]['adequate'], validation_nmse=qm[e, c]['validation_nmse'], is_control=m.startswith('control_'))
        for (e, c, m), v in grouped.items()]
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        derived_field_files_sha256=cache_hashes, all_subjects=data.n, all_original_rows_in_source_fit_and_moments=data.m,
        registration_max_residual=max(errors)))
    protocol.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', protocol)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3'); p.add_argument('--batch', action='store_true')
    p.add_argument('--source-run', default='waveAM_global_output_link_seed412')
    p.add_argument('--run-name', default='waveAO_equal_X_global_link_seed412')
    args = p.parse_args(); survivors = None
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            fields = parent/args.source_run
            if not fields.exists():
                import run_global_output_link as preparation
                folders = [parent/'joint_bank_seed391', parent/'joint_bank_degree5_seed391']
                for degree, folder in zip((4, 5), folders):
                    if not folder.exists():
                        joint_bank.prepare(source, folder, degrees=(degree,), device='cuda' if model == 'CR3' else 'auto')
                preparation.analyze(source, fields, folders)
            rows = analyze(fields, parent/args.run_name)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows
                      if r['adequate'] and r['all_planes_pass'] and not r['is_control']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveAO] {model}: {len(passed)} balanced-link UX passes, {len(survivors)} common', flush=True)
            if args.batch and not survivors:
                print('[WaveAO] No identical criterion remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
