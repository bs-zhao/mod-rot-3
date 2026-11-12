"""BC: derivative-free UX support from conditional input feature moments."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import conditional_x_witness as x_witness
import conditional_ui_witness as u_witness
import conditional_ux as conditional
import ux_metric_scope as scope


@torch.no_grad()
def moments(feature, flat_targets, target_squared_norm, output_width):
    n = len(feature); cross = feature.T @ flat_targets
    squared = cross.reshape(len(cross), target_squared_norm.shape[1], output_width).square().sum((0, 2))
    diagonal = feature.square().sum(1) @ target_squared_norm
    return (squared/(n*n)).cpu().numpy(), ((squared-diagonal)/(n*(n-1))).cpu().numpy()


def analyze(source, output, bank_folder, seed=430, device='cuda', centers=16, bank_factory=profiles.MatrixBank):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous conditional kernel moment experiment: {output}')
    bank = bank_factory(source, bank_folder); data = bank.data
    output.mkdir(parents=True)
    protocol = dict(status='RUNNING', model=data.index['model'], all_subjects=data.n, all_original_rows=data.m,
        source=str(Path(source).resolve()), source_matrix_manifest_sha256=bank.protocol['manifest_sha256'],
        source_response_quality=bank.quality, seed=seed, Ui_kernel_centers=centers,
        global_output_transform=bank.protocol.get('response_transform', 'identity'),
        conditional_mean='complete polynomial degree3 plus 64 radial features; nested five-fold W-only regression',
        contexts=['none', 'constant plus centered joint-other-U radial features'],
        output_mode_relative_squared_tolerance=1e-8, device=device, precision='float64',
        input_independence_assumed=False, response_function_fitted=False, response_derivatives_used=False,
        estimators=['squared empirical feature covariance', 'diagonal-removed covariance product, clipped for support'],
        interpretation='conditional feature moment diagnostic, not exact Sobol decomposition or calibrated CI test',
        prediction_quality_policy='continuous diagnostic, no fixed MSE veto',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_conditional_kernel_moments.py', 'conditional_ui_witness.py', 'conditional_x_witness.py',
             'matrix_function_profiles.py', 'conditional_ux.py', 'tail_support.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    z = data.u-data.u.mean(0); z /= np.sqrt(np.var(data.u, axis=0).mean())
    maps, reveal = [], {}
    for k, path in enumerate(data.paths):
        q, _ = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); reveal[f'case_{k:03d}'] = path.parent.name
    directions, membership = conditional.direction_classes(maps)
    np.savez_compressed(output/'W_only_geometry.npz', directions=directions, membership=membership)
    operators, audit = x_witness.operators(bank.x)
    wave.json_file(output/'X_feature_audit.json', audit)
    for j, operator in enumerate(operators):
        np.save(output/f'X_witness_column{j}.npy', operator, allow_pickle=False)
    tasks, targets, discarded, mode_audit = profiles.task_modes(bank.responses, operators, tolerance=1e-8)
    wave.json_file(output/'matrix_function_modes.json', dict(tasks=mode_audit,
        maximum_discarded_relative_energy=float(np.max(discarded.sum(1)/data.n)),
        moment_norm_absolute_error_bound='sqrt(discarded task energy), because Ui features have unit empirical mean squared norm'))
    width = max(t.shape[1] for t in targets); values = np.zeros((data.n, len(tasks), width))
    for t, target in enumerate(targets):
        values[:, t, :target.shape[1]] = target
    flat = torch.as_tensor(values.reshape(data.n, -1), dtype=torch.float64, device=device)
    norm = torch.as_tensor(np.sum(values*values, axis=-1), dtype=torch.float64, device=device)
    graphs, audits = {}, []
    for i, direction in enumerate(directions):
        features, audit = u_witness.features(z, direction, seed=seed, centers=centers)
        audits.append(dict(direction=i, **audit))
        for view, feature in features.items():
            raw, corrected = moments(torch.as_tensor(feature, dtype=torch.float64, device=device), flat, norm, width)
            for estimator, scores in (('raw', raw), ('diagonal_removed', corrected)):
                key = view+'_'+estimator
                if key not in graphs:
                    graphs[key] = np.zeros((len(data.channels), len(directions), data.dx))
                for t, (c, j) in enumerate(tasks):
                    graphs[key][c, i, j] = scores[t]
        wave.json_file(output/'U_feature_audit.json', audits)
        if (i+1) % 4 == 0 or i+1 == len(directions):
            print(f'[WaveBC] {data.index["model"]}: W-only conditional Ui witnesses {i+1}/{len(directions)}', flush=True)
    np.savez_compressed(output/'signed_conditional_moments.npz', **graphs)
    wave.json_file(output/'negative_moment_diagnostics.json', {k:dict(negative_fraction=float(np.mean(v < 0.)),
        minimum=float(v.min()), maximum=float(v.max())) for k, v in graphs.items()})
    blind = []
    for view, graph in graphs.items():
        for k, axes in enumerate(membership):
            for c, channel in enumerate(data.channels):
                for metric, value in conditional.sparsity(np.maximum(graph[c, axes], 0.)).items():
                    blind.append(dict(anonymous_case=f'case_{k:03d}', estimator='conditional_kernel_moment',
                        channel=channel, metric=view+'_'+metric, loss=value))
    wave.csv_file(output/'blind_scores.csv', blind)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = [dict(estimator='conditional_kernel_moment', channel=c, metric=m,
        **wave.st.evaluate_curve(values, data.index['cases']), is_control=not scope.primary(m),
        prediction_error_veto_applied=False) for (c, m), values in grouped.items()]
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_subjects=data.n, all_original_rows=data.m, source_matrix_manifest_sha256=bank.protocol['manifest_sha256']))
    protocol.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', protocol)
    print(f'[WaveBC] {data.index["model"]}: '
          f'{sum(r["all_planes_pass"] and not r["is_control"] for r in rows)} primary grid combinations', flush=True)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3'); p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveBC_conditional_kernel_UX_seed430')
    p.add_argument('--centers', type=int, default=16)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args()
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            folder = parent/'matrix_response_table_seed383'
            if not folder.exists():
                profiles.prepare(source, folder)
            analyze(source, parent/args.run_name, folder, device=args.device, centers=args.centers)
