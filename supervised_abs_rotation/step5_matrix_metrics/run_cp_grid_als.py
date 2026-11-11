"""Wave AF: higher-rank UX function compression with exact-grid ALS."""
import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import cp_regression as cp
import cp_grid_als as als
import approximate_ux_function as teacher_functions
import fitted_bank
import prepare_response_bank
import joint_bank
import ux_predictive_complexity as uc


def prepare_grid(data, maps, output, teacher, seed, device):
    common = teacher_functions.shared_x(data)
    if teacher == 'poly4_UX':
        raw, audit = teacher_functions.polynomial(data, 4)
    elif teacher == 'rbf2_UX':
        raw, audit = teacher_functions.kernel(data, 'rbf2')
    else:
        raise ValueError('An approximated UX function is required')
    if common is not None:
        nodes = common
        target = raw.reshape(data.n, len(nodes), len(data.channels)).transpose(2, 0, 1)
        reference = 'complete original shared-X grid; all original subject/trial rows'
    else:
        folder = data.folder.parent/'step5_matrix_metrics'/'UX_complexity_source_seed383'
        if not folder.exists():
            prepare_response_bank.prepare(data.folder, folder, estimators=('poly3', 'poly5', 'rbf1', 'rbf2'))
        bank = fitted_bank.FittedBank(data.folder, folder)
        with np.load(bank.folder/'response_bank.npz', allow_pickle=False) as saved:
            nodes = saved['x'].copy()
        if teacher == 'rbf2_UX':
            coefficients, z, meta = uc.source_coefficients(bank, 'rbf2')
            phi = uc.wave.sur.monomials(z, meta['powers'], np.ones(len(meta['powers']))) if meta['kind'] == 'poly' else None
            if phi is None:
                ds = np.maximum(np.sum(z*z, 1)[:, None]+np.sum(meta['centers']**2, 1)[None]-2*z @ meta['centers'].T, 0.)
                phi = np.column_stack((np.ones(len(z)), np.exp(-ds/(2*meta['lengthscale']**2))))
            target = (phi @ coefficients.reshape(len(phi.T), -1)).reshape(data.n, len(data.channels), len(nodes)).transpose(1, 0, 2)
        else:
            source = joint_bank.JointBank(data.folder, data.folder.parent/'step5_matrix_metrics'/'joint_bank_seed391')
            saved = source.load(4)
            up, xp = wave.sur.powers_for(data.d, 4), wave.sur.powers_for(data.dx, 4)
            ul = {tuple(p): i for i, p in enumerate(up)}; xl = {tuple(p): i for i, p in enumerate(xp)}
            tensor = np.zeros((len(up), len(data.channels), len(xp)))
            for p, w, c in zip(saved['powers'], saved['weights'], saved['coef']):
                tensor[ul[tuple(p[:data.d])], :, xl[tuple(p[data.d:])]] += w*c
            u = (source.data.u-saved['mean'][:data.d])/saved['scale'][:data.d]
            x = (nodes-saved['mean'][data.d:])/saved['scale'][data.d:]
            uphi = wave.sur.monomials(u, up, np.ones(len(up)))
            xphi = wave.sur.monomials(x, xp, np.ones(len(xp)))
            coefficients = (uphi @ tensor.reshape(len(up), -1)).reshape(data.n, len(data.channels), len(xp))
            target = np.einsum('ncq,mq->cnm', coefficients, xphi, optimize=True)
        reference = 'all empirical U subjects times observed joint-X source-bank nodes; source fit uses all original rows'
    target = target-target.mean(1, keepdims=True)-target.mean(2, keepdims=True)+target.mean((1, 2), keepdims=True)
    sd = np.sqrt(np.mean(target**2, axis=(1, 2)))
    if np.any(sd < 1e-12):
        raise ValueError('No identifiable coupling variation')
    target /= sd[:, None, None]
    umean = data.u.mean(0); uscale = np.sqrt(np.var(data.u, axis=0).mean())
    u = (data.u-umean)/uscale
    classes, membership = cp.canonical_classes(u, maps)
    up, um, ut, au = cp.whiten(cp.features(np.einsum('nd,gde->gne', u, classes)))
    scaler = wave.sur.Scaler(data.w[:, data.d:], 1)
    xp, xm, xt, ax = cp.whiten(cp.features(scaler(nodes)))
    np.savez_compressed(output/'feature_geometry.npz', U_mean=umean, U_scale=uscale,
        canonical_maps_from_W=classes, case_membership=membership, U_basis_mean=um,
        U_basis_transform=ut, U_basis_active=au, X_mean=scaler.mean, X_scale=scaler.scale,
        X_basis_mean=xm, X_basis_transform=xt, X_basis_active=ax, X_nodes=nodes, coupling_scale=sd)
    audit.update(compression_reference=reference, compression_subjects=data.n, compression_X_nodes=len(nodes),
        grid_rows=data.n*len(nodes), all_original_rows_in_source_fit=data.m,
        score_equals_original_trial_grid=common is not None)
    wave.json_file(output/'teacher_audit.json', audit)
    convert = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)
    return dict(u_phi=convert(up), x_phi=convert(xp), active_u=convert(au), active_x=convert(ax),
        targets=convert(target), groups=len(classes), channels=len(data.channels)), membership, audit


def fit_rank(arrays, output, rank, sweeps, starts, seed, previous=None, model_chunk=8):
    model = als.GridALS(**arrays, starts=starts, rank=rank, seed=seed, previous=previous)
    best = model.loss(model_chunk); state = model.snapshot()
    history = [dict(sweep=0, loss=best.cpu().tolist())]; previous_loss = best.clone()
    elapsed = time.perf_counter()
    for sweep in range(1, sweeps+1):
        for block, dimensions in (('u', model.u.shape[1]), ('x', model.x.shape[1])):
            for axis in range(dimensions):
                model.update(block, axis, ridge=1e-8, model_chunk=model_chunk)
        current = model.loss(model_chunk)
        improved = current < best
        best[improved] = current[improved]
        snapshot = model.snapshot()
        for key in state:
            state[key][improved] = snapshot[key][improved]
        if sweep % 5 == 0 or sweep == 1 or sweep == sweeps:
            direct = (model.predict_grid(0)-model.targets[model.channel_index[0]]).square().mean()
            if abs(float(direct-current[0])) > 1e-7*max(1., float(direct)):
                raise ArithmeticError('Exact grid loss failed direct full-grid prediction audit')
            history.append(dict(sweep=sweep, loss=current.cpu().tolist(), best_loss=best.cpu().tolist(),
                seconds=time.perf_counter()-elapsed, direct_grid_loss_check=float(abs(direct-current[0])),
                maximum_amplitude=float(model.amplitude.max()),
                maximum_increase_over_previous_sweep=float((current-previous_loss).max())))
            wave.json_file(output/f'optimization_rank{rank}.json', dict(history=history, latest_sweep=sweep,
                ridge=1e-8, current_mean_relative_mse=float(current.mean()), best_mean_relative_mse=float(best.mean())))
            torch.save(dict(best_state=state, losses=best, rank=rank, latest_sweep=sweep), output/f'best_rank{rank}.pt')
            print(f'[WaveAF] rank {rank} sweep {sweep}/{sweeps}: mean {current.mean().item():.6g}, '
                  f'best {best.mean().item():.6g}, max {current.max().item():.6g}, '
                  f'{time.perf_counter()-elapsed:.1f}s', flush=True)
        previous_loss = current
    losses = best.cpu().numpy().reshape(starts, arrays['channels'], arrays['groups'])
    return state, losses


def analyze(source, output, teacher='poly4_UX', ranks=(1, 2, 4, 8, 16), sweeps=40,
            starts=3, seed=408, device='cuda', model_chunk=8):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain prior grid ALS experiment: {output}')
    data = wave.io.MatrixCases(source, ('logit_nn', 'p_nn'), seed)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(source.resolve()),
        seed=seed, teacher=teacher, ranks=ranks, sweeps=sweeps, starts=starts, device=device,
        precision='float64 throughout', channels=data.channels, score_sign='lower for every dataset',
        basis='whitened 1,z,z2,z3,z4 and five RBF knots -2..2 per coordinate',
        reference='empirical whole-U times observed joint-X blocks; scalar-coordinate dependence retained',
        all_subjects_used=True, all_original_rows_in_source_fit=True, intercept='free scalar only',
        solver='independent per-model alternating linear least squares, profiled intercept, ridge 1e-8',
        status='RUNNING', optimization_stability_verified=False,
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_cp_grid_als.py', 'cp_grid_als.py', 'cp_regression.py', 'approximate_ux_function.py',
             'joint_bank.py', 'fitted_bank.py', 'product_response.py', 'matrix_io.py', 'surrogates.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    maps, reveal, errors = [], {}, []
    for k, path in enumerate(data.paths):
        q, error = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); errors.append(error); reveal[f'case_{k:03d}'] = path.parent.name
    arrays, membership, audit = prepare_grid(data, maps, output, teacher, seed, device)
    previous, all_losses = None, {}
    for rank in ranks:
        previous, losses = fit_rank(arrays, output, rank, sweeps, starts, seed, previous, model_chunk)
        all_losses[rank] = losses
        np.savez_compressed(output/'anonymous_fit_losses.npz', **{f'rank{r}': v for r, v in all_losses.items()})
    blind = []
    for c, channel in enumerate(data.channels):
        by_rank = np.asarray([all_losses[r][:, c].min(0) for r in ranks])
        for k, group in enumerate(membership):
            metrics = {f'rank{r}_UX_prediction_tail_l2': float(np.sqrt(by_rank[j, group])) for j, r in enumerate(ranks)}
            for eps in (.01, .03, .1, .3):
                hits = np.flatnonzero(by_rank[:, group] <= eps**2)
                metrics[f'UX_budget_count_eps{int(eps*1000):03d}'] = int(ranks[hits[0]]) if len(hits) else max(ranks)+1
            for metric, loss in metrics.items():
                blind.append(dict(anonymous_case=f'case_{k:03d}', estimator='CP_ALS_basis10',
                    channel=channel, metric=metric, loss=loss))
    wave.csv_file(output/'blind_scores.csv', blind)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = [dict(estimator=e, channel=c, metric=m, **wave.st.evaluate_curve(v, data.index['cases']),
        adequate=audit['validation_nmse'][data.channels.index(c)] <= .1,
        optimization_stability_verified=False, is_control=False) for (e, c, m), v in grouped.items()]
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        source_function_audit='teacher_audit.json', all_subjects=data.n,
        all_original_rows_in_source_fit=data.m, scored_grid_rows=audit['grid_rows'],
        exact_original_grid=audit['score_equals_original_trial_grid'], registration_max_residual=max(errors)))
    protocol.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', protocol)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3'); p.add_argument('--batch', action='store_true')
    p.add_argument('--teacher', choices=('poly4_UX', 'rbf2_UX'), default='poly4_UX')
    p.add_argument('--run-name', default='waveAF_poly4_UX_ALS_seed408')
    p.add_argument('--ranks', default='1,2,4,8,16'); p.add_argument('--sweeps', type=int, default=40)
    p.add_argument('--starts', type=int, default=3); p.add_argument('--model-chunk', type=int, default=8)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args(); ranks = tuple(map(int, args.ranks.split(','))); survivors = None
    if tuple(sorted(set(ranks))) != ranks or min(ranks) < 1 or min(args.sweeps, args.starts, args.model_chunk) < 1:
        p.error('Increasing positive ranks and positive execution counts are required')
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model)
            rows = analyze(source, source.parent/'step5_matrix_metrics'/args.run_name, args.teacher,
                ranks, args.sweeps, args.starts, device=args.device, model_chunk=args.model_chunk)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows if r['adequate'] and r['all_planes_pass']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveAF] {model}: {len(passed)} preliminary grid passes, {len(survivors)} common', flush=True)
            if args.batch and not survivors:
                print('[WaveAF] No identical criterion remains; optimization stability still requires audit', flush=True)
                return


if __name__ == '__main__':
    main()
