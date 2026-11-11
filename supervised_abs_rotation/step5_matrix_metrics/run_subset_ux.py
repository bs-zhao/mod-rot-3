"""Wave AH: minimum-U-subset complexity of fitted UX interaction functions."""
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
import subset_ux as subset
import ux_metric_scope as scope


def analyze(source, output, bank_folder, estimators=('poly5', 'rbf2'), degree=5, centers=128, seed=407, folds=5):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain prior subset UX experiment: {output}')
    bank = fitted_bank.FittedBank(source, bank_folder); data = bank.data
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(Path(source).resolve()),
        seed=seed, source_manifest_sha256=bank.source_digest, estimators=estimators, channels=data.channels,
        conditional_degree=degree, RBF_centers=centers, RBF_lengthscale=2., U_crossfit_folds=folds,
        score_sign='lower for every dataset', all_subjects_used=True, all_original_rows_in_source_fit=True,
        primary_scope='minimum U-variable support for each omitted-X coupling contrast, relative prediction tails',
        subset_selection='best pooled out-of-fold error per subset size; descriptive compression cost, not nested-wrapper test error',
        adequacy='source validation NMSE <= .1 and full-U crossfit relative MSE <= .1 for every X contrast',
        conditional_reference='observed whole-U and joint-X blocks; no scalar-coordinate independence',
        status='RUNNING', code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_subset_ux.py', 'subset_ux.py', 'conditional_ux.py', 'ux_predictive_complexity.py',
             'fitted_bank.py', 'matrix_io.py', 'surrogates.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    maps, reveal, registration_errors = [], {}, []
    for k, path in enumerate(data.paths):
        q, error = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); registration_errors.append(error); reveal[f'case_{k:03d}'] = path.parent.name
    bases, memberships = subset.subspace_classes(maps)
    np.savez_compressed(output/'W_only_subspaces.npz', **{f'basis{k}': b for k, b in enumerate(bases)})
    wave.json_file(output/'W_only_memberships.json', memberships)
    cached = output.parent/'waveZ_UX_complexity_seed405'
    if cached.exists():
        old = json.loads((cached/'protocol.json').read_text())
        digest = hashlib.sha256((cached/'blind_scores.csv').read_bytes()).hexdigest()
        if old['status'] != 'COMPLETE' or digest != old['scores_sha256'] or old['source_signature_sha256'] != bank.source_digest:
            raise ValueError('Source X contrasts do not match this fitted function bank')
        operators = {f'omit_d{order}': [load_npz(cached/f'X_omit_d{order}_column{j}.npz').toarray()
                     for j in range(data.dx)] for order in (2, 4)}
        wave.json_file(output/'X_map_audit.json', dict(frozen_source=str(cached.resolve()), scores_sha256=digest))
    else:
        with np.load(bank.folder/'response_bank.npz', allow_pickle=False) as saved:
            x = saved['x']
        scale = wave.sur.Scaler(data.w[:, data.d:], 1).scale
        operators, audit = uc.omission_operators(x, scale, bank.bank, bank.train, seed=405)
        wave.json_file(output/'X_map_audit.json', audit)
    blind, quality = [], []
    for name in estimators:
        coefficients, z, meta = uc.source_coefficients(bank, name)
        latents, weights, grams, audit = conditional.balanced_latents(conditional.source_features(z, meta), coefficients, operators)
        wave.json_file(output/f'{name}_latent_audit.json', audit)
        np.savez_compressed(output/f'{name}_latent_geometry.npz', latents=latents, loss_weights=weights,
                            **{mode: g for mode, g in grams.items()})
        normalizers = {mode: np.trace(g, axis1=-2, axis2=-1) for mode, g in grams.items()}
        ratios = {mode: [] for mode in grams}; histories = []
        for k, basis in enumerate(bases):
            covariance, fit = conditional.omission_covariance(z, None, latents, weights, degree=degree,
                centers=centers, seed=seed, folds=folds, basis=basis)
            for mode, gram in grams.items():
                loss = np.einsum('ab,cxab->cx', covariance, gram, optimize=True)
                ratios[mode].append(loss/np.maximum(normalizers[mode], 1e-30))
            histories.append(dict(subspace=k, dimension=basis.shape[1], **fit))
            if k % 8 == 0 or k+1 == len(bases):
                wave.json_file(output/f'{name}_crossfit_audit.json', histories)
                np.savez_compressed(output/f'{name}_subspace_relative_errors.npz', **{m: np.asarray(v) for m, v in ratios.items()})
                print(f'[WaveAH] {data.index["model"]} {name}: subspaces {k+1}/{len(bases)}', flush=True)
        full = next(k for k, b in enumerate(bases) if b.shape[1] == data.d)
        for mode, values in ratios.items():
            values = np.asarray(values)
            curves = subset.budget_curves(values, memberships, data.d)
            np.save(output/f'{name}_{mode}_budget_curves.npy', curves)
            for c, channel in enumerate(data.channels):
                qr = bank.quality[name, channel]
                worst = float(values[full, c].max())
                quality.append(dict(estimator=name, channel=channel, X_mode=mode,
                    adequate=qr['adequate'] == 'True' and worst <= .1,
                    validation_nmse=float(qr['validation_nmse']), worst_full_U_crossfit_relative_mse=worst))
                for case in range(len(maps)):
                    for metric, loss in subset.scores(curves[case, :, c]).items():
                        blind.append(dict(anonymous_case=f'case_{case:03d}', estimator=name, channel=channel,
                                          metric=mode+'_'+metric, loss=loss))
        wave.csv_file(output/'blind_scores.csv', blind); wave.csv_file(output/'fit_quality.csv', quality)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    groups = defaultdict(dict); qm = {(r['estimator'], r['channel'], r['X_mode']): r for r in quality}
    for row in blind:
        groups[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = []
    for (name, channel, metric), losses in groups.items():
        mode = next(m for m in operators if metric.startswith(m+'_')); qr = qm[name, channel, mode]
        rows.append(dict(estimator=name, channel=channel, metric=metric,
            **wave.st.evaluate_curve(losses, data.index['cases']), adequate=qr['adequate'],
            validation_nmse=qr['validation_nmse'], full_U_relative_mse=qr['worst_full_U_crossfit_relative_mse'],
            is_control=not scope.primary(metric)))
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)),
        derived_function_files=sorted(bank.derived_reads), forbidden_files_read=[], all_subjects=data.n,
        all_original_rows_in_source_fit=data.m, registration_max_residual=max(registration_errors)))
    protocol.update(status='COMPLETE', scores_sha256=digest, unique_W_subspaces=len(bases))
    wave.json_file(output/'protocol.json', protocol)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3'); p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveAH_minimum_UX_subset_seed407')
    p.add_argument('--centers', type=int, default=128)
    args = p.parse_args(); survivors = None
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            bank = parent/'UX_complexity_source_seed383'
            if not bank.exists():
                prepare_response_bank.prepare(source, bank, estimators=('poly3', 'poly5', 'rbf1', 'rbf2'))
            rows = analyze(source, parent/args.run_name, bank, centers=args.centers)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows
                      if r['adequate'] and r['all_planes_pass'] and not r['is_control']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveAH] {model}: {len(passed)} adequate UX subset passes, {len(survivors)} common', flush=True)
            if args.batch and not survivors:
                print('[WaveAH] No identical criterion remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
