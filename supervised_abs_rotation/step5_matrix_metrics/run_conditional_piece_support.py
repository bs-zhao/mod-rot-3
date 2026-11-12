"""Wave AS: conditional active directions and monotone pieces of UX effects."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import load_npz
from threadpoolctl import threadpool_limits

import run as wave
import fitted_bank
import prepare_response_bank
import ux_predictive_complexity as uc
import cp_regression as cp
import conditional_piece_support as pieces


def analyze(source, output, bank_folder, estimators=('poly5', 'rbf2'), nodes=17, neighbors=64,
            subjects_per_chunk=32, seed=418, device='cuda', effect_view='X_omission'):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous conditional piece experiment: {output}')
    bank = fitted_bank.FittedBank(source, bank_folder); data = bank.data
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(Path(source).resolve()),
        seed=seed, source_seed=bank.protocol['seed'], source_manifest_sha256=bank.source_digest,
        estimators=estimators, channels=data.channels, nodes=nodes, conditional_neighbors=neighbors,
        interval_conditional_quantiles=[.1, .9], support_k=16, support_factor=1.5, minimum_probe_coverage=.9,
        effect_view=effect_view,
        primary_scope=('conditional U monotonicity of raw response at each X context' if effect_view == 'raw_response' else
                       'soft count of active UX directions and active monotone pieces, varying one Ui with other U fixed'),
        effect_energy='squared slice range / squared interval width',
        activation=('e/(e+eps^2 * sum_U e) independently in every X context' if effect_view == 'raw_response' else
                    'e/(e+eps^2 * mean_X_context(sum_U e)) within each subject and X-column contrast'),
        piece_count='slice total variation / slice range', relative_thresholds=[.01, .03, .1],
        all_subjects_used=True, all_original_rows_in_source_fit=data.m, all_source_X_contexts_used=bank.bank.shape[2],
        device=device, precision='float64', score_sign='lower for every dataset', status='RUNNING',
        prediction_quality_policy='continuous diagnostic, no fixed MSE veto in continuation',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_conditional_piece_support.py', 'conditional_piece_support.py', 'ux_predictive_complexity.py',
             'cp_regression.py', 'fitted_bank.py', 'matrix_io.py', 'surrogates.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    maps, reveal, errors = [], {}, []
    for k, path in enumerate(data.paths):
        q, error = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); errors.append(error); reveal[f'case_{k:03d}'] = path.parent.name
    cached = output.parent/'waveZ_UX_complexity_seed405'; reuse = False
    if cached.exists():
        old = json.loads((cached/'protocol.json').read_text())
        digest = hashlib.sha256((cached/'blind_scores.csv').read_bytes()).hexdigest()
        reuse = old['status'] == 'COMPLETE' and digest == old['scores_sha256'] and old['source_signature_sha256'] == bank.source_digest
    if effect_view == 'raw_response':
        operators = None
        wave.json_file(output/'X_crossfit_audit.json', dict(X_omission_performed=False,
            reason='evaluate raw response monotonicity with original X held fixed; no X contrast'))
    elif effect_view != 'X_omission':
        raise ValueError(f'Unknown effect view: {effect_view}')
    elif reuse:
        operators = [load_npz(cached/f'X_omit_d2_column{j}.npz').toarray() for j in range(data.dx)]
        wave.json_file(output/'X_crossfit_audit.json', dict(source_run=str(cached.resolve()), scores_sha256=digest))
    else:
        with np.load(bank.folder/'response_bank.npz', allow_pickle=False) as saved:
            x = saved['x']
        mapping, audit = uc.omission_operators(x, wave.sur.Scaler(data.w[:, data.d:], 1).scale,
            bank.bank, bank.train, seed=405, degrees=(2,))
        operators = mapping['omit_d2']; wave.json_file(output/'X_crossfit_audit.json', audit)
    operators = np.asarray(operators) if operators is not None else None
    blind, quality, coverage = [], [], []
    for name in estimators:
        coefficients, z, meta = uc.source_coefficients(bank, name)
        if effect_view == 'raw_response':
            contrasts = coefficients[:, :, None, :]
            score_dx, reference = 1, 'context'
        else:
            coefficients -= coefficients.mean(2, keepdims=True)
            contrasts = np.einsum('pcm,jkm->pcjk', coefficients, operators, optimize=True)
            score_dx, reference = data.dx, 'field'
        evaluator = pieces.Evaluator(meta, contrasts.reshape(len(coefficients), -1), device=device)
        classes, membership = cp.canonical_classes(z, maps)
        np.savez_compressed(output/f'{name}_W_only_classes.npz', classes=classes, membership=membership)
        tables_by_class, audits = [], []
        for group, q in enumerate(classes):
            tables, audit = pieces.evaluate_case(z, q, evaluator, len(data.channels), score_dx, bank.bank.shape[2],
                nodes=nodes, neighbors=neighbors, subjects_per_chunk=subjects_per_chunk, reference=reference)
            if effect_view == 'raw_response':
                for key in list(tables):
                    if key.startswith('eps'):
                        tables['reversal_'+key] = np.maximum(tables[key]-tables['active_'+key], 0.)
            tables_by_class.append(tables); audits.append(dict(estimator=name, coordinate_class=group, **audit))
            np.savez_compressed(output/f'{name}_subject_costs_class{group:03d}.npz', **tables)
            wave.json_file(output/f'{name}_probe_coverage.json', audits)
            print(f'[WaveAS] {data.index["model"]} {name}: conditional slices {group+1}/{len(classes)}, '
                  f'coverage {audit["probe_coverage"]:.4f}', flush=True)
        minimum = min(row['probe_coverage'] for row in audits)
        coverage.extend(audits)
        for c, channel in enumerate(data.channels):
            old = bank.quality[name, channel]
            quality.append(dict(estimator=name, channel=channel, adequate=old['adequate'] == 'True' and minimum >= .9,
                validation_nmse=float(old['validation_nmse']), minimum_probe_coverage=minimum))
            for case, group in enumerate(membership):
                for metric, value in pieces.scores(tables_by_class[group], c).items():
                    if effect_view == 'raw_response':
                        if '_Xworst_' in metric:
                            continue
                        metric = metric.replace('UX_conditional_', 'U_raw_conditional_').replace('_Xmean_', '_')
                        if 'reversal_' in metric:
                            metric = metric.replace('monotone_pieces_', 'monotonicity_excess_').replace('reversal_', '')
                    blind.append(dict(anonymous_case=f'case_{case:03d}', estimator=name, channel=channel, metric=metric, loss=value))
        wave.csv_file(output/'blind_scores.csv', blind); wave.csv_file(output/'fit_quality.csv', quality)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict); qm = {(r['estimator'], r['channel']): r for r in quality}
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = [dict(estimator=e, channel=c, metric=m, **wave.st.evaluate_curve(v, data.index['cases']),
        adequate=qm[e, c]['adequate'], validation_nmse=qm[e, c]['validation_nmse'], is_control=False)
        for (e, c, m), v in grouped.items()]
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        derived_function_files=sorted(bank.derived_reads), all_subjects=data.n, all_original_rows_in_source_fit=data.m,
        all_source_X_contexts_scored=bank.bank.shape[2], registration_max_residual=max(errors)))
    protocol.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', protocol)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3'); p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveAS_conditional_UX_pieces_seed418')
    p.add_argument('--nodes', type=int, default=17); p.add_argument('--neighbors', type=int, default=64)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--effect-view', choices=('X_omission', 'raw_response'), default='X_omission')
    args = p.parse_args(); survivors = None
    if args.nodes < 3 or args.neighbors < 2:
        p.error('At least three nodes and two neighbors are required')
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            folder = parent/'UX_complexity_source_seed383'
            if folder.exists() and not {'poly5', 'rbf2'} <= set(json.loads((folder/'protocol.json').read_text())['estimators']):
                folder = parent/'UX_piece_source_seed383'
            if not folder.exists():
                prepare_response_bank.prepare(source, folder, estimators=('poly5', 'rbf2'))
            rows = analyze(source, parent/args.run_name, folder, nodes=args.nodes, neighbors=args.neighbors,
                           device=args.device, effect_view=args.effect_view)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows if r['all_planes_pass']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveAS] {model}: {len(passed)} conditional UX complexity passes, {len(survivors)} common', flush=True)
            if args.batch and not survivors:
                print('[WaveAS] No identical adequate criterion remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
