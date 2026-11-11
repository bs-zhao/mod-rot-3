"""Wave AL: sparse U-polynomial terms with arbitrary observed-X profiles."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import load_npz
from threadpoolctl import threadpool_limits

import run as wave
import fitted_bank
import prepare_response_bank
import ux_predictive_complexity as uc
import conditional_ux as conditional
import fast_projection as fp
from all_trial_moments import FrozenScaler
import profile_polynomial_complexity as profile


def analyze(source, output, bank_folder, estimators=('poly5', 'rbf2'), degrees=(5, 7), maximum=256):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain prior profile-complexity experiment: {output}')
    bank = fitted_bank.FittedBank(source, bank_folder); data = bank.data
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(Path(source).resolve()),
        source_seed=bank.protocol['seed'], source_manifest_sha256=bank.source_digest,
        estimators=estimators, U_projection_degrees=degrees, maximum=maximum, channels=data.channels,
        all_subjects_used_in_fit_and_compression=True, all_original_rows_in_source_fit=True,
        primary_scope='few U polynomial terms with unrestricted observed-X profiles, per-X relative L2 tails',
        X_profile_function_form_assumed=False, compression_reference='observed whole U and joint-X blocks',
        source_latent_relative_squared_error_tolerance=1e-7, full_polynomial_relative_MSE_adequacy=.1,
        score_sign='lower for every dataset', status='RUNNING',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_profile_polynomial_complexity.py', 'profile_polynomial_complexity.py', 'conditional_ux.py',
             'ux_predictive_complexity.py', 'fast_projection.py', 'fitted_bank.py', 'matrix_io.py', 'surrogates.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    maps, reveal, errors = [], {}, []
    for k, path in enumerate(data.paths):
        q, error = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); errors.append(error); reveal[f'case_{k:03d}'] = path.parent.name
    cached = output.parent/'waveZ_UX_complexity_seed405'
    if cached.exists():
        old = json.loads((cached/'protocol.json').read_text())
        digest = hashlib.sha256((cached/'blind_scores.csv').read_bytes()).hexdigest()
        if old['status'] != 'COMPLETE' or old['scores_sha256'] != digest or old['source_signature_sha256'] != bank.source_digest:
            raise ValueError('Frozen X contrasts do not match this source')
        operators = {f'omit_d{degree}': [load_npz(cached/f'X_omit_d{degree}_column{j}.npz').toarray()
                     for j in range(data.dx)] for degree in (2, 4)}
        wave.json_file(output/'X_map_audit.json', dict(source=str(cached.resolve()), scores_sha256=digest))
    else:
        with np.load(bank.folder/'response_bank.npz', allow_pickle=False) as saved:
            x = saved['x']
        operators, audit = uc.omission_operators(x, wave.sur.Scaler(data.w[:, data.d:], 1).scale, bank.bank, bank.train, seed=405)
        wave.json_file(output/'X_map_audit.json', audit)
    blind, quality = [], []
    for name in estimators:
        coefficients, z, meta = uc.source_coefficients(bank, name)
        latents, _, grams, audit = conditional.balanced_latents(conditional.source_features(z, meta), coefficients, operators)
        wave.json_file(output/f'{name}_source_function_audit.json', audit)
        for degree in degrees:
            model = fp.Polynomial(FrozenScaler(np.zeros(data.d), np.ones(data.d)), degree)
            histories, full_errors = [], defaultdict(list)
            for case, q in enumerate(maps):
                phi = model.features(z @ q)[:, 1:]
                phi -= phi.mean(0); gram = phi.T @ phi/data.n; cross = phi.T @ latents/data.n
                for mode, tasks in grams.items():
                    for c, channel in enumerate(data.channels):
                        curves = []
                        for j, task in enumerate(tasks[c]):
                            target_gram = cross @ task @ cross.T
                            energy = float(np.trace(task))
                            full, rank = profile.full_error(gram, target_gram, energy)
                            full_errors[mode, c].append(full)
                            curve, fit = profile.compress(gram, target_gram, energy, maximum=maximum)
                            curves.append(curve)
                            histories.append(dict(anonymous_case=f'case_{case:03d}', X_mode=mode, channel=channel,
                                X_column=j, full_projection_relative_mse=full, full_effective_rank=rank,
                                relative_error_curve=curve.tolist(), **fit))
                        for metric, loss in profile.scores(curves, maximum).items():
                            blind.append(dict(anonymous_case=f'case_{case:03d}', estimator=name,
                                channel=channel, metric=f'Udegree{degree}_{mode}_'+metric, loss=loss))
                wave.csv_file(output/'blind_scores.csv', blind)
                wave.json_file(output/f'{name}_Udegree{degree}_compression_audit.json', histories)
                print(f'[WaveAL] {data.index["model"]} {name} Udegree{degree}: cases {case+1}/{len(maps)}', flush=True)
            for mode in operators:
                for c, channel in enumerate(data.channels):
                    values = np.asarray(full_errors[mode, c]).reshape(len(maps), data.dx)
                    drift = float(np.ptp(values, axis=0).max())
                    if drift > 1e-5:
                        raise ArithmeticError(f'Full U polynomial function space changed under rotation: {drift}')
                    qr = bank.quality[name, channel]
                    quality.append(dict(estimator=name, channel=channel, U_degree=degree, X_mode=mode,
                        adequate=qr['adequate'] == 'True' and float(values.max()) <= .1,
                        validation_nmse=float(qr['validation_nmse']), worst_full_projection_relative_mse=float(values.max()),
                        full_projection_rotation_drift=drift))
            wave.csv_file(output/'fit_quality.csv', quality)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict); qm = {(r['estimator'], r['channel'], r['U_degree'], r['X_mode']): r for r in quality}
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = []
    for (name, channel, metric), values in grouped.items():
        degree = int(metric.split('_')[0][7:]); mode = next(m for m in operators if metric.startswith(f'Udegree{degree}_{m}_'))
        qr = qm[name, channel, degree, mode]
        rows.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(values, data.index['cases']), adequate=qr['adequate'],
            validation_nmse=qr['validation_nmse'], full_U_projection_mse=qr['worst_full_projection_relative_mse'], is_control=False))
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        derived_function_files=sorted(bank.derived_reads), all_subjects_in_compression=data.n,
        all_original_rows_in_source_fit=data.m, registration_max_residual=max(errors)))
    protocol.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', protocol)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3'); p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveAL_UX_profile_terms_seed411')
    args = p.parse_args(); survivors = None
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            bank = parent/'UX_complexity_source_seed383'
            if not bank.exists():
                prepare_response_bank.prepare(source, bank, estimators=('poly3', 'poly5', 'rbf1', 'rbf2'))
            rows = analyze(source, parent/args.run_name, bank)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows if r['adequate'] and r['all_planes_pass']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveAL] {model}: {len(passed)} profile-complexity passes, {len(survivors)} common', flush=True)
            if args.batch and not survivors:
                print('[WaveAL] No identical criterion remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
