"""Rescore one frozen blind frame for every kernel support/tail diagnostic."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import learned_response_profiles as learned
import conditional_ui_witness as ui
import conditional_ux as conditional
import run_conditional_kernel_moments as kernel
import ux_metric_scope as scope


def analyze(source, original, folder, counterexample, output, learned_link=False, device='cuda'):
    original, counterexample, output = map(Path, (original, counterexample, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain earlier kernel family audit: {output}')
    protocol = json.loads((original/'protocol.json').read_text())
    if protocol['status']!='COMPLETE' or hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()!=protocol['scores_sha256']:
        raise ValueError('Kernel source changed or is incomplete')
    frozen = json.loads((counterexample/'blind_freeze.json').read_text())
    rotation_digest = hashlib.sha256((counterexample/'blind_rotations.npz').read_bytes()).hexdigest()
    if frozen['rotations_sha256']!=rotation_digest:
        raise ValueError('Counterexample frame changed')
    with np.load(counterexample/'blind_rotations.npz', allow_pickle=False) as saved:
        q = saved['optimized'].copy()
    bank = (learned.MatrixBank if learned_link else profiles.MatrixBank)(source, folder)
    if bank.protocol['manifest_sha256']!=protocol['source_matrix_manifest_sha256']:
        raise ValueError('Response source changed')
    output.mkdir(parents=True)
    data = bank.data; z = data.u-data.u.mean(0); z /= np.sqrt(np.var(data.u, axis=0).mean())
    operators = [np.load(original/f'X_witness_column{j}.npy', allow_pickle=False) for j in range(data.dx)]
    tasks, targets, _, _ = profiles.task_modes(bank.responses, operators, tolerance=1e-8)
    width = max(t.shape[1] for t in targets); values = np.zeros((data.n, len(tasks), width))
    for t, target in enumerate(targets):
        values[:, t, :target.shape[1]] = target
    flat = torch.tensor(values.reshape(data.n, -1), dtype=torch.float64, device=device)
    norm = torch.tensor(np.sum(values*values, axis=-1), dtype=torch.float64, device=device)
    graphs, audits = {}, []
    for i, v in enumerate(q.T):
        features, audit = ui.features(z, v, seed=protocol['seed'], centers=protocol['Ui_kernel_centers'])
        audits.append(dict(axis=i, **audit))
        for view, feature in features.items():
            raw, corrected = kernel.moments(torch.tensor(feature, dtype=torch.float64, device=device), flat, norm, width)
            for estimator, scores in (('raw', raw), ('diagonal_removed', corrected)):
                key = view+'_'+estimator
                if key not in graphs:
                    graphs[key] = np.zeros((len(data.channels), data.d, data.dx))
                for t, (c, j) in enumerate(tasks):
                    graphs[key][c, i, j] = scores[t]
    rows = []
    for view, graph in graphs.items():
        for c, channel in enumerate(data.channels):
            rows.extend(dict(channel=channel, metric=view+'_'+m, loss=value, is_control=not scope.primary(m))
                        for m, value in conditional.sparsity(np.maximum(graph[c], 0.)).items())
    wave.csv_file(output/'blind_frame_scores.csv', rows)
    digest = hashlib.sha256((output/'blind_frame_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'blind_freeze.json', dict(scores_sha256=digest, rotations_sha256=rotation_digest))
    np.savez_compressed(output/'signed_moments.npz', **graphs)
    wave.json_file(output/'new_U_feature_audits.json', audits)
    # Only now join the frozen source's labelled evaluation summaries.
    with (original/'metric_summary.csv').open() as stream:
        summaries = {(r['channel'],r['metric']):r for r in csv.DictReader(stream)}
    reveal = json.loads((original/'reveal.json').read_text())['mapping']
    baseline_case = next(k for k,v in reveal.items() if v=='baseline')
    with (original/'blind_scores.csv').open() as stream:
        baselines = {(r['channel'],r['metric']):float(r['loss']) for r in csv.DictReader(stream)
                     if r['anonymous_case']==baseline_case}
    assessment = []
    for row in rows:
        summary = summaries[row['channel'],row['metric']]; baseline = baselines[row['channel'],row['metric']]
        assessment.append(dict(**row, baseline_loss=baseline,
            relative_blind_margin=(row['loss']-baseline)/max(abs(baseline), 1e-30),
            grid_planes_passed=summary['success_planes'], grid_planes_total=summary['n_planes'],
            grid_baseline_rank=summary['baseline_rank']))
    wave.csv_file(output/'assessment.csv', assessment)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[]))
    wave.json_file(output/'protocol.json', dict(status='COMPLETE', source_scores_sha256=protocol['scores_sha256'],
        scores_sha256=digest, all_subjects=data.n, all_original_rows=data.m,
        new_optimization=False, original_response_transform=protocol.get('global_output_transform', 'identity')))
    print(f'[Kernel family transfer] {len(rows)} variants frozen; no new optimization', flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--learned-link', action='store_true')
    args = p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        source = wave.io.default_step2(wave.ROOT,'CR3'); parent = source.parent/'step5_matrix_metrics'
        original = parent/('waveBF_learned_global_link_kernel_seed430' if args.learned_link else
                           'waveBC2_conditional_kernel_centers32_seed430')
        folder = parent/('matrix_response_global_warp_learned_seed438' if args.learned_link else 'matrix_response_table_seed383')
        tag = 'BF' if args.learned_link else 'BC2'
        analyze(source, original, folder, parent/'waveBC3_conditional_kernel_blind_seed439',
                parent/f'wave{tag}_frozen_counter_family_v2', args.learned_link)
