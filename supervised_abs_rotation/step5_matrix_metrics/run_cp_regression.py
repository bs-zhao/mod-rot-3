"""Wave Y: GPU separated-rank regression budgets on all original W/Y rows."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import fast_projection as fp
import cp_regression as cp


def prepare_arrays(data, maps, output, device, seed):
    rng = np.random.default_rng(seed)
    train = np.sort(rng.permutation(data.n)[:int(.75*data.n)])
    x = data.w[:, data.d:]
    scaler = wave.sur.Scaler(x, 1, row_mask=np.isin(data.sid, train))
    nuisance = fp.Polynomial(scaler, 4)
    coef, fit = fp.fit(nuisance, x, data.y, data.sid, train, device=device)
    wave.json_file(output/'X_nuisance_fit.json', fit)
    np.savez_compressed(output/'X_nuisance_model.npz', coef=coef, mean=scaler.mean, scale=scaler.scale,
                        powers=nuisance.powers, weights=nuisance.weights)
    residual = np.empty_like(data.y)
    for a in range(0, data.m, 8192):
        b = min(a+8192, data.m)
        residual[a:b] = data.y[a:b]-nuisance.features(x[a:b]) @ coef
    center = residual.mean(0); sd = residual.std(0)
    fraction = sd**2/np.maximum(data.y.var(0), np.finfo(float).tiny)
    if np.any(fraction <= 1e-8):
        raise ValueError('Insufficient residual response variation for an axis-complexity claim')
    target = (residual-center)/sd
    mean_u = data.u.mean(0); scale_u = np.sqrt(np.var(data.u, axis=0).mean())
    u = (data.u-mean_u)/scale_u
    classes, membership = cp.canonical_classes(u, maps)
    transformed = np.einsum('nd,gde->gne', u, classes)
    up, um, ut, au = cp.whiten(cp.features(transformed))
    counts = np.diff(data.offsets); shared = bool(np.all(counts == counts[0]))
    if shared:
        common = np.asarray(x[:counts[0]])
        for a, b in zip(data.offsets[:-1], data.offsets[1:]):
            if not np.array_equal(x[a:b], common):
                shared = False; break
    raw_x = common if shared else x
    xp, xm, xt, ax = cp.whiten(cp.features(scaler(raw_x)))
    np.savez_compressed(output/'feature_geometry.npz', U_mean=mean_u, U_scale=scale_u,
        canonical_maps_from_W=classes, case_membership=membership,
        U_basis_mean=um, U_basis_transform=ut, U_basis_active=au,
        X_mean=scaler.mean, X_scale=scaler.scale, X_basis_mean=xm, X_basis_transform=xt,
        X_basis_active=ax, response_residual_mean=center, response_residual_scale=sd)
    to_device = lambda a, dtype: torch.as_tensor(a, dtype=dtype, device=device)
    arrays = dict(u=to_device(up, torch.float64), x=to_device(xp, torch.float64),
        active_u=to_device(au, torch.float32), active_x=to_device(ax, torch.float32),
        target=to_device(target, torch.float64), sid=to_device(data.sid, torch.long),
        shared_x=shared, x_count=len(xp), n_rows=data.m, n_subjects=data.n, groups=len(classes),
        membership=membership, d=data.d, dx=data.dx, channels=len(data.channels))
    audit = dict(all_subjects=data.n, all_original_rows=data.m, coordinate_classes=len(classes),
        original_cases=len(maps), class_grouping_uses_case_labels=False,
        signed_permutation_tolerance=1e-5, remaining_response_variance_fraction=fraction.tolist(),
        shared_X_cache_numerically_verified=shared,
        U_feature_ranks=au.sum(-1).tolist(), X_feature_ranks=ax.sum(-1).tolist())
    wave.json_file(output/'feature_audit.json', audit)
    return arrays


def x_rows(arrays, rows):
    return arrays['x'][rows % arrays['x_count'] if arrays['shared_x'] else rows]


@torch.no_grad()
def evaluate(model, arrays, batch, model_chunk):
    device = model.u.device; n = arrays['n_rows']; count = len(model.u)
    total = torch.zeros(count, dtype=torch.float64, device=device)
    for a in range(0, n, batch):
        rows = torch.arange(a, min(a+batch, n), device=device)
        phi = x_rows(arrays, rows); subjects = arrays['sid'][rows]
        for start in range(0, count, model_chunk):
            ids = torch.arange(start, min(start+model_chunk, count), device=device)
            prediction = model.predict(ids, subjects, phi, arrays['u'], arrays['active_u'], arrays['active_x'], torch.float64)
            target = arrays['target'][rows][:, model.channel_index[ids]].T
            total[ids] += (prediction-target).square().sum(1)
    return total/n


def fit_rank(arrays, output, rank, epochs=80, starts=3, batch=8192, model_chunk=32, seed=404, previous=None):
    device = arrays['u'].device
    model = cp.Model(arrays['groups'], arrays['channels'], starts, arrays['d'], arrays['dx'],
                     arrays['u'].shape[-1], rank, device=device, seed=seed, previous=previous)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    count = len(model.u); best_loss = evaluate(model, arrays, batch, model_chunk)
    best_state = model.snapshot(); history = [dict(epoch=0, losses=best_loss.cpu().tolist())]
    print(f'[WaveY] rank {rank} initialized: {count} independent fits; mean relative MSE {best_loss.mean().item():.6g}', flush=True)
    for epoch in range(1, epochs+1):
        lr = .01 if epoch <= 40 else (.003 if epoch <= 60 else .001)
        for group in optimizer.param_groups:
            group['lr'] = lr
        generator = torch.Generator(device=device).manual_seed(seed+rank*1000+epoch)
        order = torch.randperm(arrays['n_rows'], generator=generator, device=device)
        total = torch.zeros(count, device=device)
        for a in range(0, arrays['n_rows'], batch):
            rows = order[a:a+batch]; subjects = arrays['sid'][rows]; phi = x_rows(arrays, rows)
            optimizer.zero_grad(set_to_none=True)
            for first in range(0, count, model_chunk):
                ids = torch.arange(first, min(first+model_chunk, count), device=device)
                prediction = model.predict(ids, subjects, phi, arrays['u'], arrays['active_u'], arrays['active_x'])
                target = arrays['target'][rows][:, model.channel_index[ids]].T.to(torch.float32)
                loss = (prediction-target).square().mean(1)
                if not torch.isfinite(loss).all():
                    raise ArithmeticError('Nonfinite CP fit; preserve checkpoint, do not score as a metric failure')
                regularized = loss+1e-6*model.amplitude[ids].square().sum(1)
                regularized.sum().backward()
                total[ids] += loss.detach()*len(rows)
            cp.clip_independent_gradients(model)
            optimizer.step()
        epoch_loss = total/arrays['n_rows']
        if epoch in (10, 40, 80) or epoch == epochs:
            measured = evaluate(model, arrays, batch, model_chunk)
            improve = measured < best_loss
            best_loss[improve] = measured[improve]
            current = model.snapshot()
            for key in best_state:
                best_state[key][improve] = current[key][improve]
            history.append(dict(epoch=epoch, losses=measured.cpu().tolist(), best_losses=best_loss.cpu().tolist()))
        if epoch % 5 == 0 or epoch == epochs:
            torch.save(dict(rank=rank, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                best_state=best_state, best_loss=best_loss, seed=seed, history=history), output/f'checkpoint_rank{rank}.pt')
            wave.json_file(output/f'optimization_rank{rank}.json', dict(history=history,
                latest_epoch=epoch, moving_epoch_mean_loss=float(epoch_loss.mean()),
                moving_epoch_max_loss=float(epoch_loss.max()), all_original_rows_seen_per_epoch=arrays['n_rows']))
            print(f'[WaveY] rank {rank}, epoch {epoch}/{epochs}: moving mean MSE {epoch_loss.mean().item():.6g}, max {epoch_loss.max().item():.6g}', flush=True)
    torch.save(dict(best_state=best_state, losses=best_loss, rank=rank), output/f'best_rank{rank}.pt')
    losses = best_loss.detach().cpu().numpy().reshape(starts, arrays['channels'], arrays['groups'])
    return best_state, losses


def analyze(source, output, ranks=(1, 2, 4, 8), epochs=80, starts=3, batch=8192, model_chunk=32, seed=404, device='cuda'):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain existing CP experiment: {output}')
    data = wave.io.MatrixCases(source, ('logit_nn', 'p_nn'), seed)
    output.mkdir(parents=True, exist_ok=True)
    config = dict(model=data.index['model'], Z_dim=data.d, n_subjects=data.n, source=str(source.resolve()), seed=seed,
        ranks=list(ranks), epochs=epochs, starts=starts, batch=batch, model_chunk=model_chunk, device=device,
        channels=list(data.channels), univariate_basis='1,z,z2,z3 and five symmetric RBFs at -2,-1,0,1,2',
        factor_normalization='empirical marginal L2; centered whitened nonconstant features',
        ridge_output_amplitude_penalty=1e-6, gradient_clip_per_independent_model=5.,
        learning_rates={'1-40': .01, '41-60': .003, '61+': .001},
        nuisance='all-data X-only polynomial4; ridge selected by subject validation',
        metric_scope='fixed separated-rank empirical approximation budget; optimization validation still required',
        all_subjects_used=True, all_original_rows_used_in_fit_and_metric=True,
        training_precision='float32', final_loss_precision='float64', score_sign='lower for every dataset',
        status='RUNNING', code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('run_cp_regression.py', 'cp_regression.py', 'fast_projection.py', 'matrix_io.py', 'surrogates.py', 'structure.py'))).hexdigest())
    wave.json_file(output/'protocol.json', config)
    maps, reveal, residuals = [], {}, []
    for k, path in enumerate(data.paths):
        q, residual = wave.io.registration(data.u, data.coordinates(path))
        maps.append(q); residuals.append(residual); reveal[f'case_{k:03d}'] = path.parent.name
    arrays = prepare_arrays(data, maps, output, device, seed)
    all_losses, previous = {}, None
    for rank in ranks:
        previous, losses = fit_rank(arrays, output, rank, epochs, starts, batch, model_chunk, seed, previous)
        all_losses[rank] = losses
        np.savez_compressed(output/'anonymous_fit_losses.npz', **{f'rank{r}': v for r, v in all_losses.items()})
    blind, tables = [], {}
    for c, channel in enumerate(data.channels):
        by_rank = np.array([all_losses[r][:, c].min(0) for r in ranks])
        for k, membership in enumerate(arrays['membership']):
            row = {}
            for j, rank in enumerate(ranks):
                row[f'rank{rank}_prediction_tail_l2'] = float(np.sqrt(by_rank[j, membership]))
            for eps in (.01, .03, .1, .3):
                hit = np.flatnonzero(by_rank[:, membership] <= eps**2)
                row[f'first_budget_count_eps{int(round(eps*1000)):03d}'] = int(ranks[hit[0]]) if len(hit) else max(ranks)+1
            for metric, value in row.items():
                blind.append(dict(anonymous_case=f'case_{k:03d}', estimator='separated_CP_basis9', channel=channel,
                                  metric=metric, loss=value))
    wave.csv_file(output/'blind_scores.csv', blind)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal))
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['estimator'], row['channel'], row['metric']][reveal[row['anonymous_case']]] = row['loss']
    rows = [dict(estimator=e, channel=c, metric=m, **wave.st.evaluate_curve(v, data.index['cases']),
                 adequate=True, is_control=False, optimization_stability_verified=False)
            for (e, c, m), v in grouped.items()]
    wave.csv_file(output/'metric_summary.csv', rows)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_subjects=data.n, all_original_rows_in_fit_and_metric=data.m, registration_max_residual=max(residuals)))
    config.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', config)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveY_cp_budget_seed404')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--starts', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=8192)
    p.add_argument('--model-chunk', type=int, default=32)
    p.add_argument('--ranks', default='1,2,4,8')
    p.add_argument('--device', choices=('cuda', 'cpu', 'numpy'), default='cuda')
    args = p.parse_args(); ranks = tuple(map(int, args.ranks.split(',')))
    if tuple(sorted(set(ranks))) != ranks or min(ranks) < 1 or min(args.epochs, args.starts, args.batch_size, args.model_chunk) < 1:
        p.error('Increasing positive ranks and positive execution counts required')
    if args.device == 'numpy':
        p.error('CP training requires Torch cpu or cuda')
    survivors = None
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); output = source.parent/'step5_matrix_metrics'/args.run_name
            rows = analyze(source, output, ranks, args.epochs, args.starts, args.batch_size, args.model_chunk, device=args.device)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows if r['all_planes_pass']}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveY] {model}: {len(passed)} preliminary grid passes; {len(survivors)} common fits', flush=True)
            if args.batch and not survivors:
                print('[WaveY] Fixed-budget grid has no survivor; convergence/resolution audit remains separate', flush=True)
                return


if __name__ == '__main__':
    main()
