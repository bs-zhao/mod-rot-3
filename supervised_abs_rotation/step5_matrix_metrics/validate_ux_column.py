"""Validate the fixed poly4/logit UX-column median 10%-tail grid lead."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment, minimize
from threadpoolctl import threadpool_limits

import run as wave
import joint_bank as jb
import joint_fields as jf
import ux_objective as ux
import compare_prepared_runs as comparison

METRIC = 'UX_columns_subject_median_fractional_count_eps010'


def hux_fields(data, saved, contexts):
    ids = jf.observed_rows(data, contexts)
    _, _, _, powers, weights, coefficients = jf.derivative_coefficients(
        saved['powers'], saved['weights'], saved['coef'])
    coefficients = coefficients[:, :data.d, data.d:, 0]
    out = []
    for a in range(0, len(ids), 4096):
        z = (np.asarray(data.w[ids[a:a+4096]], dtype=float)-saved['mean'])/saved['scale']
        out.append(wave.sur.monomials(z, powers, weights) @ coefficients.reshape(len(powers), -1))
    return np.concatenate(out).reshape(data.n, contexts, data.d, data.dx), ids


def haar(rng, d):
    q, r = np.linalg.qr(rng.normal(size=(d, d)))
    q = q*np.sign(np.diag(r))[None, :]
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q


def blind_optimize(objective, output, rng, starts, random_rotations):
    """Only function fields enter; no case labels, baseline or angles."""
    d = objective.d; results, matrices, random = [], [], []
    for _ in range(random_rotations):
        q = haar(rng, d)
        random.append(dict(loss=objective.values_gradient(q)[0], q=q.tolist()))
    for i in range(starts):
        initial = np.eye(d) if i == 0 else haar(rng, d)
        trace = []
        def target(theta):
            loss, gradient = ux.parameter_objective(theta, initial, objective)
            trace.append(float(loss))
            return loss, gradient
        result = minimize(target, np.zeros(d*(d-1)//2), method='BFGS', jac=True,
                          options=dict(maxiter=350, gtol=1e-7))
        q = initial @ expm(ux.skew_parameters(result.x, d))
        loss = objective.values_gradient(q)[0]
        matrices.append(q)
        results.append(dict(start=i, loss=loss, success=bool(result.success), message=str(result.message),
                            iterations=int(result.nit), evaluations=int(result.nfev), trace=trace, q=q.tolist()))
        wave.json_file(output/'blind_optimizer.json', dict(starts=results, random_rotations=random,
                       label_inputs_used=False))
        print(f'[UX validation] blind start {i+1}/{starts}: {loss:.9g}; {result.message}', flush=True)
    return matrices, results, random


def bootstrap_scores(values_by_case, baseline, interior_cases, replicates, seed):
    rng = np.random.default_rng(seed); n = len(values_by_case[baseline])
    ids = rng.integers(0, n, size=(replicates, n), dtype=np.int32)
    scores = {}
    for case, values in values_by_case.items():
        parts = []
        for a in range(0, replicates, 20):
            parts.append(np.median(values[ids[a:a+20]], axis=1).mean(axis=1))
        scores[case] = np.concatenate(parts)
    minimum = np.min(np.stack([scores[c]-scores[baseline] for c in interior_cases]), axis=0)
    return dict(replicates=replicates, seed=seed, method='paired subject bootstrap, frozen response fit',
        simultaneous_minimum_margin_quantiles={str(p): float(np.quantile(minimum, p)) for p in (.01, .05, .5, .95, .99)},
        fraction_all_interior_margins_positive=float(np.mean(minimum > 1e-7)),
        fit_uncertainty_included=False)


def analyze(source, output, bank_folder, source_run, contexts=32, starts=16, random_rotations=128, bootstrap=400, seed=398):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain prior validation: {output}')
    prior_folder = source.parent/'step5_matrix_metrics'/source_run
    prior = json.loads((prior_folder/'protocol.json').read_text())
    if prior['status'] != 'COMPLETE' or hashlib.sha256((prior_folder/'blind_scores.csv').read_bytes()).hexdigest() != prior['scores_sha256']:
        raise ValueError('Frozen source screen is incomplete or modified')
    with (prior_folder/'metric_summary.csv').open(newline='') as stream:
        lead = [r for r in csv.DictReader(stream) if (r['estimator'], r['channel'], r['metric']) == ('poly4', 'logit_nn', METRIC)]
    if len(lead) != 1 or lead[0]['adequate'] != 'True' or lead[0]['all_planes_pass'] != 'True':
        raise ValueError('Fixed lead did not pass this dataset; no validation-success claim')
    bank = jb.JointBank(source, bank_folder); data = bank.data
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(source.resolve()),
        estimator='poly4', channel='logit_nn', metric=METRIC, contexts=contexts, starts=starts,
        random_rotations=random_rotations, bootstrap_replicates=bootstrap, seed=seed,
        source_run=source_run, source_scores_sha256=prior['scores_sha256'],
        source_bank_signature_sha256=bank.protocol['manifest_sha256'],
        status='RUNNING', all_subjects_used=True, optimized_without_labels=True,
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('validate_ux_column.py', 'ux_objective.py', 'joint_fields.py', 'joint_bank.py',
             'matrix_io.py', 'surrogates.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    hux, ids = hux_fields(data, bank.load(4), contexts)
    np.save(output/'evaluation_row_ids.npy', ids)
    np.savez_compressed(output/'hux.npz', hux=hux)
    objective = ux.UXObjective(hux)
    matrices, optimized, random = blind_optimize(objective, output, np.random.default_rng(seed), starts, random_rotations)
    optimizer_digest = hashlib.sha256((output/'blind_optimizer.json').read_bytes()).hexdigest()
    # Only now introduce the case registrations/labels for validation.
    blind, reveal, subject_values, maps = [], {}, {}, {}
    for k, path in enumerate(data.paths):
        q, _ = wave.io.registration(data.u, data.coordinates(path))
        case = f'case_{k:03d}'; reveal[case] = path.parent.name; maps[path.parent.name] = q
        values = objective.values_gradient(q, with_gradient=False)
        subject_values[path.parent.name] = values
        blind.append(dict(anonymous_case=case, estimator='poly4', channel='logit_nn', metric=METRIC,
                          loss=float(np.median(values, axis=0).mean())))
    wave.csv_file(output/'blind_scores.csv', blind)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, optimizer_sha256=optimizer_digest, mapping=reveal))
    values = {reveal[r['anonymous_case']]: r['loss'] for r in blind}
    curve = wave.st.evaluate_curve(values, data.index['cases'])
    baseline = values['baseline']; alignment = []
    for item, q in zip(optimized, matrices):
        a = np.abs(maps['baseline'].T @ q); rows, columns = linear_sum_assignment(-a)
        angles = np.degrees(np.arccos(np.clip(a[rows, columns], 0., 1.)))
        alignment.append(dict(start=item['start'], score_delta_from_baseline=item['loss']-baseline,
            mean_axis_angle_deg=float(angles.mean()), max_axis_angle_deg=float(angles.max()),
            numerical_optimizer_success=item['success']))
    interior = [c['id'] for c in data.index['cases'] if c.get('pair_1based') and c['angle_deg'] != 90]
    boot = bootstrap_scores(subject_values, 'baseline', interior, bootstrap, seed)
    assessment = dict(grid_resolution=curve, subject_bootstrap=boot, optimizer_alignment=alignment,
        baseline_loss=baseline, best_blind_loss=min(r['loss'] for r in optimized),
        random_rotation_minimum_loss=min(r['loss'] for r in random),
        resolution_and_bootstrap_pass=bool(curve['all_planes_pass'] and
            boot['simultaneous_minimum_margin_quantiles']['0.05'] > 1e-7),
        universal_success_declared=False, approximation_resolution_refit_verified=False)
    wave.json_file(output/'assessment.json', assessment)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        optimizer_sha256_before_label_evaluation=optimizer_digest, all_subjects=data.n,
        original_rows_used_in_projection_fit=data.m, observed_rows_evaluated=len(ids)))
    protocol.update(status='COMPLETE', scores_sha256=digest, optimizer_sha256=optimizer_digest)
    wave.json_file(output/'protocol.json', protocol)
    return assessment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--source-run', default='waveR_local_joint_seed397')
    p.add_argument('--run-name', default='waveS_ux_validation_seed398')
    p.add_argument('--contexts', type=int, default=32)
    p.add_argument('--starts', type=int, default=16)
    p.add_argument('--random-rotations', type=int, default=128)
    p.add_argument('--bootstrap', type=int, default=400)
    args = p.parse_args()
    if min(args.contexts, args.starts, args.random_rotations, args.bootstrap) < 1:
        p.error('Positive counts required')
    if args.batch:
        paths = {m: wave.io.default_step2(wave.ROOT, m).parent/'step5_matrix_metrics'/args.source_run for m in wave.io.DIMS}
        summary = comparison.compare(paths, wave.ROOT/'outputs'/'supervised_abs_rotation'/f'step5_{args.source_run}_four_models.csv')
        if not any(r['all_four_grid_pass'] and r['metric'] == METRIC and r['estimator'] == 'poly4' and r['channel'] == 'logit_nn' for r in summary):
            raise ValueError('The prospectively fixed lead is not a common four-model grid candidate')
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            result = analyze(source, parent/args.run_name, parent/'joint_bank_seed391', args.source_run,
                             args.contexts, args.starts, args.random_rotations, args.bootstrap)
            print(f'[UX validation] {model}: resolution/subject-bootstrap pass {result["resolution_and_bootstrap_pass"]}', flush=True)
            if args.batch and not result['resolution_and_bootstrap_pass']:
                print('[UX validation] Stability failure retained; later models skipped', flush=True)
                return


if __name__ == '__main__':
    main()
