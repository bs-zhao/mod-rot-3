"""Preserve exact-grid CP fits and extend their rank budgets prospectively."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import run_cp_grid_als as grid


def analyze(base, output, ranks=(32, 64), sweeps=40, device='cuda', model_chunk=8):
    base, output = Path(base), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous rank extension: {output}')
    old = json.loads((base/'protocol.json').read_text())
    digest = hashlib.sha256((base/'blind_scores.csv').read_bytes()).hexdigest()
    if old['status'] != 'COMPLETE' or digest != old['scores_sha256']:
        raise ValueError('Only extend a complete, frozen rank experiment')
    if min(ranks) <= max(old['ranks']) or tuple(sorted(set(ranks))) != tuple(ranks):
        raise ValueError('Rank extensions must be strictly increasing above the existing maximum')
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(old, status='RUNNING', extension_source=str(base.resolve()), source_scores_sha256=digest,
        original_ranks=old['ranks'], ranks=list(old['ranks'])+list(ranks), extension_sweeps=sweeps,
        extension_device=device, extension_code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('extend_cp_grid_als.py', 'run_cp_grid_als.py', 'cp_grid_als.py', 'cp_regression.py', 'structure.py'))).hexdigest())
    protocol.pop('scores_sha256', None)
    wave.json_file(output/'protocol.json', protocol)
    data = wave.io.MatrixCases(old['source'], tuple(old['channels']), old['seed'])
    maps, reveal, errors = [], {}, []
    for k, path in enumerate(data.paths):
        q, error = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); errors.append(error); reveal[f'case_{k:03d}'] = path.parent.name
    arrays, membership, teacher = grid.prepare_grid(data, maps, output, old['teacher'], old['seed'], device)
    checks = {}
    with np.load(base/'feature_geometry.npz', allow_pickle=False) as before, np.load(output/'feature_geometry.npz', allow_pickle=False) as after:
        if set(before.files) != set(after.files):
            raise ValueError('Feature geometry schema changed')
        for key in before.files:
            if before[key].shape != after[key].shape:
                raise ValueError('Feature geometry shape changed')
            maximum = float(np.max(np.abs(before[key].astype(float)-after[key].astype(float))))
            if maximum > 1e-8*max(1., float(np.max(np.abs(before[key].astype(float))))):
                raise ArithmeticError(f'Cannot transfer rank checkpoint after feature geometry changed: {key}')
            checks[key] = maximum
    maximum_rank = max(old['ranks'])
    checkpoint = torch.load(base/f'best_rank{maximum_rank}.pt', map_location=device, weights_only=True)
    previous = checkpoint['best_state']
    with np.load(base/'anonymous_fit_losses.npz', allow_pickle=False) as saved:
        losses = {r: saved[f'rank{r}'].copy() for r in old['ranks']}
    np.testing.assert_allclose(checkpoint['losses'].cpu().numpy().reshape(losses[maximum_rank].shape), losses[maximum_rank], atol=1e-12)
    wave.json_file(output/'resume_audit.json', dict(feature_geometry_maximum_differences=checks,
        original_scores_sha256=digest, reused_artifacts_sha256={f: hashlib.sha256((base/f).read_bytes()).hexdigest()
            for f in ('protocol.json', 'feature_geometry.npz', 'anonymous_fit_losses.npz', f'best_rank{maximum_rank}.pt')},
        original_low_rank_fits_unchanged=True))
    for rank in ranks:
        previous, value = grid.fit_rank(arrays, output, rank, sweeps, old['starts'], old['seed'], previous, model_chunk)
        losses[rank] = value
        np.savez_compressed(output/'anonymous_fit_losses.npz', **{f'rank{r}': v for r, v in losses.items()})
    all_ranks = list(old['ranks'])+list(ranks); blind = []
    for c, channel in enumerate(data.channels):
        by_rank = np.asarray([losses[r][:, c].min(0) for r in all_ranks])
        for k, group in enumerate(membership):
            metrics = {f'rank{r}_UX_prediction_tail_l2': float(np.sqrt(by_rank[j, group])) for j, r in enumerate(all_ranks)}
            for eps in (.01, .03, .1, .3):
                hits = np.flatnonzero(by_rank[:, group] <= eps**2)
                metrics[f'UX_budget_count_eps{int(eps*1000):03d}'] = int(all_ranks[hits[0]]) if len(hits) else max(all_ranks)+1
            for metric, loss in metrics.items():
                blind.append(dict(anonymous_case=f'case_{k:03d}', estimator='CP_ALS_basis10', channel=channel, metric=metric, loss=loss))
    wave.csv_file(output/'blind_scores.csv', blind)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = [dict(estimator=e, channel=c, metric=m, **wave.st.evaluate_curve(v, data.index['cases']),
        adequate=teacher['validation_nmse'][data.channels.index(c)] <= .1,
        optimization_stability_verified=False, is_control=False) for (e, c, m), v in grouped.items()]
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        source_function_audit='teacher_audit.json', all_subjects=data.n, all_original_rows_in_source_fit=data.m,
        scored_grid_rows=teacher['grid_rows'], exact_original_grid=teacher['score_equals_original_trial_grid'],
        registration_max_residual=max(errors)))
    protocol.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', protocol)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3'); p.add_argument('--batch', action='store_true')
    p.add_argument('--teacher', choices=('poly4_UX', 'rbf2_UX'), default='poly4_UX')
    p.add_argument('--source-run', default='waveAF_poly4_UX_ALS_seed408')
    p.add_argument('--run-name', default='waveAI_poly4_UX_ALS_rank64_seed408')
    p.add_argument('--ranks', default='32,64'); p.add_argument('--sweeps', type=int, default=40)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args(); ranks = tuple(map(int, args.ranks.split(','))); survivors = None
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            base = parent/args.source_run
            if not base.exists():
                grid.analyze(source, base, teacher=args.teacher, device=args.device)
            previous = json.loads((base/'protocol.json').read_text())
            if previous['teacher'] != args.teacher or previous['ranks'] != [1, 2, 4, 8, 16] or previous['sweeps'] != 40 or previous['starts'] != 3:
                raise ValueError('All datasets must use the prescribed identical source-rank protocol')
            rows = analyze(base, parent/args.run_name, ranks, args.sweeps, args.device)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows if r['adequate'] and r['all_planes_pass']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveAI/AJ] {model}: {len(passed)} preliminary rank-budget passes, {len(survivors)} common', flush=True)
            if args.batch and not survivors:
                print('[WaveAI/AJ] No identical criterion remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
