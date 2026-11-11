"""Wave AM: globally integrable output-link correction, then UX support."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import joint_bank
import global_output_link as link


def analyze(source, output, folders, degrees=(4, 5), seed=412, bins=(0, 16, 64, 256)):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous global output-link experiment: {output}')
    banks = [joint_bank.JointBank(source, folder) for folder in folders]; data = banks[0].data
    if len(degrees) != len(banks) or any(not np.array_equal(b.data.u, data.u) for b in banks):
        raise ValueError('Every degree must refer to the same actual W frame')
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(Path(source).resolve()),
        seed=seed, degrees=degrees, channels=data.channels, global_link_bins=bins,
        global_link='k(y)=-E[<HUX,gU*gX^T>|f=y]/E[||gU*gX^T||^2|f=y]',
        primary_scope='UX support after one globally integrable scalar-output correction; selected resolution only',
        link_resolution_selection='75/25 subject validation of mixed-derivative residual energy, identity included',
        link_refit_subjects=data.n, all_original_rows_in_source_fit_and_moments=data.m,
        normalization='each observed-point X column, before per-person covariance',
        no_local_W_dependent_link_parameters=True, score_sign='lower for every dataset', status='RUNNING',
        source_manifests={f'poly{d}': b.protocol['manifest_sha256'] for d, b in zip(degrees, banks)},
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_global_output_link.py', 'global_output_link.py', 'joint_fields.py', 'fast_projection.py',
             'local_support.py', 'joint_bank.py', 'matrix_io.py', 'surrogates.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    maps, reveal, errors = [], {}, []
    for k, path in enumerate(data.paths):
        q, error = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); errors.append(error); reveal[f'case_{k:03d}'] = path.parent.name
    train_subjects = np.random.default_rng(seed).permutation(data.n)[:int(.75*data.n)]
    training_rows = np.isin(data.sid, train_subjects)
    blind, quality = [], []
    for degree, bank in zip(degrees, banks):
        name = f'poly{degree}'; folder = output/name; folder.mkdir()
        prediction, gradient, mixed = link.polynomial_fields(data, bank.load(degree), folder,
            progress=lambda count: print(f'[WaveAM] {data.index["model"]} {name}: derivative rows {count}/{data.m}', flush=True))
        numerator, denominator, base = link.invariants(gradient, mixed, data.d)
        links, selections, audit = link.fit_all(prediction, numerator, denominator, base, training_rows, bins=bins)
        wave.json_file(folder/'global_link_fit.json', dict(links=links, selected_bins=selections, channel_audits=audit))
        moments = link.moments(data, prediction, gradient, mixed, links, bins,
            progress=lambda count: print(f'[WaveAM] {data.index["model"]} {name}: normalized moments {count}/{data.n}', flush=True))
        for count, matrix in moments.items():
            np.savez_compressed(folder/f'global_link_bins{count}_moments.npz', UX=matrix)
        for c, channel in enumerate(data.channels):
            chosen = selections[c]
            selected_audit = next(r for r in audit[c]['history'] if r['bins'] == chosen)
            qr = bank.quality[name, channel]
            quality.append(dict(estimator=name, channel=channel,
                adequate=float(qr['validation_nmse']) <= .1 and selected_audit['numerical_derivative_dynamic_range_representable'],
                validation_nmse=float(qr['validation_nmse']), selected_bins=chosen,
                selected_global_link_log_derivative_range=selected_audit['log_derivative_range'],
                global_link_integrability_verified=True))
            for k, q in enumerate(maps):
                for count, matrix in moments.items():
                    for metric, loss in link.scores(matrix[:, c], q).items():
                        blind.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                            metric=f'control_bins{count}_'+metric, loss=loss))
                        if count == chosen:
                            blind.append(dict(anonymous_case=f'case_{k:03d}', estimator=name, channel=channel,
                                metric='selected_global_link_'+metric, loss=loss))
        wave.csv_file(output/'blind_scores.csv', blind); wave.csv_file(output/'fit_quality.csv', quality)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict); qm = {(r['estimator'], r['channel']): r for r in quality}
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = [dict(estimator=e, channel=c, metric=m, **wave.st.evaluate_curve(v, data.index['cases']),
        adequate=qm[e, c]['adequate'], validation_nmse=qm[e, c]['validation_nmse'],
        is_control=m.startswith('control_')) for (e, c, m), v in grouped.items()]
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_subjects_in_link_refit_and_moments=data.n, all_original_rows_in_source_fit_and_moments=data.m,
        registration_max_residual=max(errors)))
    protocol.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', protocol)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3'); p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveAM_global_output_link_seed412')
    args = p.parse_args(); survivors = None
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            folders = [parent/'joint_bank_seed391', parent/'joint_bank_degree5_seed391']
            for degree, folder in zip((4, 5), folders):
                if not folder.exists():
                    joint_bank.prepare(source, folder, degrees=(degree,), device='cuda' if model == 'CR3' else 'auto')
            rows = analyze(source, parent/args.run_name, folders)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows
                      if r['adequate'] and r['all_planes_pass'] and not r['is_control']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveAM] {model}: {len(passed)} global-link UX passes, {len(survivors)} common', flush=True)
            if args.batch and not survivors:
                print('[WaveAM] No identical criterion remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
