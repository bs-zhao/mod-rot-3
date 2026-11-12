"""Audit the frozen tree family over a common declared error-resolution path.

No tree is refitted and no per-model error budget is chosen. The original
10% criterion remains identified; other resolutions are exploratory checks.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import run as wave
import tree_function_complexity as trees

EPSILONS = (.05, .06, .07, .08, .09, .1, .12, .15, .2)


def analyze(source, original, output):
    original, output = Path(original), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain earlier resolution audit: {output}')
    old = json.loads((original/'protocol.json').read_text())
    digest = hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()
    if old['status'] != 'COMPLETE' or digest != old['scores_sha256']:
        raise ValueError('Original tree experiment is incomplete or changed')
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(status='RUNNING', model=old['model'], epsilons=EPSILONS,
        original_criterion_relative_L2=.1, original_scores_sha256=digest,
        source_matrix_manifest_sha256=old['source_matrix_manifest_sha256'],
        original_trees_refitted=False, all_original_subjects=old['n_subjects'],
        prediction_quality_policy='continuous diagnostic; no fixed veto',
        interpretation='common resolution sensitivity; exploratory, no per-model setting selection',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    with np.load(original/'geometry.npz', allow_pickle=False) as geometry:
        membership = geometry['membership']; classes = geometry['W_only_classes']
    with np.load(original/'per_X_complexity.npz', allow_pickle=False) as saved:
        reference = saved['counts']
    costs = np.zeros((len(classes), len(old['channels']), len(EPSILONS), reference.shape[-1]))
    path_hashes = {}
    for group in range(len(classes)):
        filename = original/f'distortion_paths_class{group:03d}.npz'
        path_hashes[filename.name] = hashlib.sha256(filename.read_bytes()).hexdigest()
        with np.load(filename, allow_pickle=False) as paths:
            for c in range(len(old['channels'])):
                for j in range(reference.shape[-1]):
                    leaves = paths[f'channel{c}_X{j}_all_subject_leaves']
                    errors = paths[f'channel{c}_X{j}_all_subject_errors'][0]
                    costs[group, c, :, j] = [trees.minimum_leaves(leaves, errors, eps) for eps in EPSILONS]
    if not np.array_equal(costs[:, :, EPSILONS.index(.1)], reference[:, :, 2]):
        raise ArithmeticError('The 10% resolution no longer reproduces the frozen tree counts')
    blind = []
    for case, group in enumerate(membership):
        for c, channel in enumerate(old['channels']):
            for k, eps in enumerate(EPSILONS):
                for pooling, loss in (('Xmean', np.mean(costs[group, c, k])), ('Xworst', np.max(costs[group, c, k]))):
                    blind.append(dict(anonymous_case=f'case_{case:03d}', channel=channel,
                        relative_L2=eps, pooling=pooling, loss=float(loss)))
    wave.csv_file(output/'blind_scores.csv', blind)
    new_digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'blind_freeze.json', dict(scores_sha256=new_digest, path_sha256=path_hashes))
    reveal = json.loads((original/'reveal.json').read_text())
    if reveal['scores_sha256'] != digest:
        raise ValueError('Original score reveal changed')
    index = json.loads((Path(source)/'index.json').read_text())
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['channel'], row['relative_L2'], row['pooling']][reveal['mapping'][row['anonymous_case']]] = row['loss']
    summary = []
    for (channel, eps, pooling), values in grouped.items():
        baseline = values['baseline']; minimum = min(values.values())
        summary.append(dict(model=old['model'], channel=channel, relative_L2=eps, pooling=pooling,
            **wave.st.evaluate_curve(values, index['cases']), baseline_leaf_count=baseline,
            minimum_leaf_count=minimum, best_observed_relative_advantage=(baseline-minimum)/baseline,
            original_criterion=eps == .1 and pooling == 'Xworst', prediction_error_veto_applied=False))
    wave.csv_file(output/'resolution_evidence.csv', summary)
    protocol.update(status='COMPLETE', scores_sha256=new_digest, original_10percent_counts_identical=True)
    wave.json_file(output/'protocol.json', protocol)
    return summary


if __name__ == '__main__':
    all_rows = []
    for model in ('CR3', 'C1', 'C2', 'C3'):
        source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
        rows = analyze(source, parent/'waveAU_conditional_UX_tree_complexity_seed421', parent/'waveAV_tree_resolution_path')
        all_rows.extend(rows)
        print(f'[WaveAV] {model}: resolution path complete; frozen 10% counts reproduced', flush=True)
    folder = wave.ROOT/'outputs/supervised_abs_rotation/waveAU_four_model_evidence'
    wave.csv_file(folder/'resolution_path_all_models.csv', all_rows)
    common = None
    for model in ('CR3', 'C1', 'C2', 'C3'):
        passing = {(r['channel'], r['relative_L2'], r['pooling']) for r in all_rows
                   if r['model'] == model and r['all_planes_pass']}
        common = passing if common is None else common & passing
    wave.json_file(folder/'resolution_common_grid_candidates.json', dict(
        same_setting_candidates=[dict(channel=c, relative_L2=e, pooling=p) for c, e, p in sorted(common)],
        original_setting_changed=False, interpretation='exploratory resolution sensitivity, not validated recovery'))
    print('[WaveAV] Common exploratory grid settings:', sorted(common), flush=True)
