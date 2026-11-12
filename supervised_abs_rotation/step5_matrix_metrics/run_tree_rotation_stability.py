"""Wave AW: continuous evidence for the frozen AU tree-complexity candidate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, linear_sum_assignment
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import tree_rotation_stability as stability
import cp_regression as cp


def analyze(source, original, output, random_rotations=64, bootstrap=24, optimizer_starts=2,
            optimizer_evaluations=80, seed=425):
    original, output = Path(original), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain earlier tree stability experiment: {output}')
    old = json.loads((original/'protocol.json').read_text())
    digest = hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()
    if old['status'] != 'COMPLETE' or old['scores_sha256'] != digest:
        raise ValueError('Only a complete immutable tree wave can be audited')
    bank = profiles.MatrixBank(source, original.parent/'matrix_response_table_seed383'); data = bank.data
    if bank.protocol['manifest_sha256'] != old['source_matrix_manifest_sha256']:
        raise ValueError('The matrix-response table differs from the frozen tree experiment')
    output.mkdir(parents=True, exist_ok=True)
    operator_files = [original/f'X_contrast_column{j}.npy' for j in range(data.dx)]
    operators = [np.load(path, allow_pickle=False) for path in operator_files]
    protocol = dict(status='RUNNING', model=data.index['model'], n_subjects=data.n,
        original_run=str(original.resolve()), original_scores_sha256=digest,
        source_matrix_manifest_sha256=bank.protocol['manifest_sha256'], seed=seed,
        channel='logit_nn', criterion='worst-X leaves for 10% relative L2 conditional function approximation',
        random_SO_rotations=random_rotations, paired_subject_bootstrap_refits=bootstrap,
        bootstrap_draw_size=data.n, optimizer_starts=optimizer_starts, optimizer_evaluations=optimizer_evaluations,
        prediction_quality_policy='report continuously; no fixed MSE veto',
        X_operator_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in operator_files],
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_tree_rotation_stability.py', 'tree_rotation_stability.py', 'tree_function_complexity.py',
             'matrix_function_profiles.py', 'cp_regression.py', 'matrix_io.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    targets, residuals = stability.coupling_modes(bank.responses, operators)
    z = data.u-data.u.mean(0); z /= np.sqrt(np.var(data.u, axis=0).mean())
    all_subjects = np.arange(data.n)
    sampled, discarded = stability.sample_functions(targets, residuals, all_subjects)
    with np.load(original/'geometry.npz', allow_pickle=False) as geometry:
        classes = geometry['W_only_classes']; membership = geometry['membership']
    actual_maps = [wave.io.registration(data.u, data.coordinates(path))[0] for path in data.paths]
    actual_classes, actual_membership = cp.canonical_classes(z, actual_maps)
    if not np.array_equal(membership, actual_membership) or not np.allclose(classes, actual_classes, atol=1e-10):
        raise ValueError('Actual case matrices no longer match the saved anonymous coordinate classes')
    # Reproduce every old W-only class before making new robustness claims.
    with np.load(original/'per_X_complexity.npz', allow_pickle=False) as saved:
        expected = saved['counts'][:, 0, 2, :]
    reproduced = []
    for k, q in enumerate(classes):
        _, counts = stability.score(z, q, sampled, discarded)
        reproduced.append(counts)
    if not np.array_equal(reproduced, expected):
        wave.json_file(output/'reproduction_failure.json', dict(expected=expected.tolist(), actual=reproduced))
        raise ArithmeticError('Fast tree audit does not reproduce the frozen per-X complexity counts')
    wave.json_file(output/'reproduction.json', dict(all_W_only_classes=len(classes), all_counts_identical=True))
    rng = np.random.default_rng(seed); blind, rotations, histories = [], [], []
    for k in range(random_rotations):
        q = stability.random_rotation(data.d, rng)
        loss, counts = stability.score(z, q, sampled, discarded)
        rotations.append(q); blind.append(dict(anonymous_rotation=k, loss=loss, per_X_leaves=counts))
        wave.csv_file(output/'blind_random_scores.csv', blind)
        if (k+1) % 8 == 0:
            print(f'[WaveAW] {data.index["model"]}: blind random rotations {k+1}/{random_rotations}', flush=True)
    best_ids = np.argsort([r['loss'] for r in blind])[:optimizer_starts]
    optimized = []
    for start, index in enumerate(best_ids):
        q0 = rotations[int(index)]
        def objective(parameters):
            q = q0 @ stability.increment(parameters, data.d)
            loss, counts = stability.score(z, q, sampled, discarded)
            histories.append(dict(start=start, evaluation=len(histories), loss=loss, per_X_leaves=counts,
                                  parameters=np.asarray(parameters).tolist()))
            wave.csv_file(output/'blind_optimization_scores.csv', histories)
            return loss
        result = minimize(objective, np.zeros(data.d*(data.d-1)//2), method='Powell',
            bounds=[(-np.pi/3, np.pi/3)]*(data.d*(data.d-1)//2),
            options=dict(maxfev=optimizer_evaluations, xtol=.01, ftol=0.))
        best_step = min((r for r in histories if r['start'] == start), key=lambda r: r['loss'])
        optimized.append(dict(rotation=q0 @ stability.increment(best_step['parameters'], data.d), loss=float(best_step['loss']),
                              success=bool(result.success), message=str(result.message), evaluations=int(result.nfev)))
        print(f'[WaveAW] {data.index["model"]}: blind optimizer {start+1}/{len(best_ids)}, loss {result.fun:g}', flush=True)
    np.savez_compressed(output/'blind_rotation_matrices.npz', random=np.asarray(rotations),
        optimized=np.asarray([r['rotation'] for r in optimized]))
    bootstrap_losses = []
    for b in range(bootstrap):
        ids = rng.integers(data.n, size=data.n)
        sampled_b, discarded_b = stability.sample_functions(targets, residuals, ids)
        losses = [stability.score(z[ids], q, sampled_b, discarded_b)[0] for q in classes]
        bootstrap_losses.append(losses)
        np.savez_compressed(output/'blind_bootstrap_scores.npz', losses=np.asarray(bootstrap_losses))
        print(f'[WaveAW] {data.index["model"]}: paired all-subject bootstrap {b+1}/{bootstrap}', flush=True)
    files = ['blind_random_scores.csv', 'blind_rotation_matrices.npz', 'blind_bootstrap_scores.npz']
    if histories:
        files.append('blind_optimization_scores.csv')
    frozen = {f: hashlib.sha256((output/f).read_bytes()).hexdigest() for f in files}
    wave.json_file(output/'blind_freeze.json', frozen)
    # Baseline, angles and axis correspondence enter only this evaluation.
    base_case = next(k for k, path in enumerate(data.paths) if path.parent.name == 'baseline')
    baseline_group = int(membership[base_case]); base_q = classes[baseline_group]
    baseline = float(np.max(expected[baseline_group]))
    bootstrap_losses = np.asarray(bootstrap_losses)
    comparisons = []
    for case, group in enumerate(membership):
        name = data.paths[case].parent.name
        margins = (bootstrap_losses[:, group]-bootstrap_losses[:, baseline_group])/np.maximum(bootstrap_losses[:, baseline_group], 1.)
        comparisons.append(dict(case=name, original_relative_margin=float((np.max(expected[group])-baseline)/max(baseline, 1.)),
            bootstrap_margin_q05=float(np.quantile(margins, .05)), bootstrap_margin_median=float(np.median(margins)),
            bootstrap_margin_q95=float(np.quantile(margins, .95)), bootstrap_fraction_strictly_above_baseline=float(np.mean(margins > 0))))
    wave.csv_file(output/'paired_bootstrap_comparisons.csv', comparisons)
    random_loss = np.array([r['loss'] for r in blind]); best = int(np.argmin(random_loss))
    alignment = base_q.T @ rotations[best]
    a, b = linear_sum_assignment(-np.abs(alignment))
    maximum_error = float(np.rad2deg(np.arccos(np.clip(np.abs(alignment[a, b]), 0., 1.))).max())
    summary = dict(baseline_loss=baseline, random_minimum_loss=float(random_loss.min()),
        random_median_loss=float(np.median(random_loss)), random_fraction_below_baseline=float(np.mean(random_loss < baseline)),
        best_random_maximum_axis_error_degrees=maximum_error,
        blind_optimizer_minimum_loss=min((r['loss'] for r in optimized), default=None),
        bootstrap_replicates=bootstrap, prediction_error_is_diagnostic=True,
        conclusion='continuous robustness evidence; no automatic acceptance or fixed-MSE veto')
    wave.json_file(output/'assessment.json', summary)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        source_matrix_manifest_sha256=bank.protocol['manifest_sha256'], all_subjects=data.n))
    protocol.update(status='COMPLETE', blind_file_sha256=frozen)
    wave.json_file(output/'protocol.json', protocol)
    print(f'[WaveAW] {data.index["model"]}: {summary}', flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--run-name', default='waveAW_tree_blind_bootstrap_seed425')
    p.add_argument('--random-rotations', type=int, default=64); p.add_argument('--bootstrap', type=int, default=24)
    p.add_argument('--optimizer-starts', type=int, default=2); p.add_argument('--optimizer-evaluations', type=int, default=80)
    args = p.parse_args()
    if args.random_rotations < 1 or args.bootstrap < 1 or args.optimizer_starts > args.random_rotations:
        p.error('Positive random rotations/bootstrap and no more optimizer starts than random rotations are required')
    source = wave.io.default_step2(wave.ROOT, args.model); parent = source.parent/'step5_matrix_metrics'
    with threadpool_limits(limits=4):
        analyze(source, parent/'waveAU_conditional_UX_tree_complexity_seed421', parent/args.run_name,
            random_rotations=args.random_rotations, bootstrap=args.bootstrap,
            optimizer_starts=args.optimizer_starts, optimizer_evaluations=args.optimizer_evaluations)


if __name__ == '__main__':
    main()
