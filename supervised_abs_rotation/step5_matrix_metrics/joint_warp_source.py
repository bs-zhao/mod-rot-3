"""Fit one global monotone output link with a GP on all actual W/Y rows."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import run as wave
import warped_gp as warp


class Dictionary:
    def __init__(self, mean, scale, centers, d, device):
        self.device, self.d = device, d
        self.mean = torch.tensor(mean, dtype=torch.float64, device=device)
        self.scale = torch.tensor(scale, dtype=torch.float64, device=device)
        center = (np.asarray(centers, dtype=float)-mean)/scale
        lengths = []
        for block in (center[:, :d], center[:, d:]):
            distance = np.maximum(np.sum(block*block, 1)[:, None]+np.sum(block*block, 1)[None]-2*block @ block.T, 0.)
            positive = distance[np.triu_indices(len(block), 1)]; positive = positive[positive > 1e-12]
            lengths.append(float(np.sqrt(np.median(positive))) if len(positive) else 1.)
        self.lengths = np.array([lengths[0]]*d+[lengths[1]]*(len(mean)-d))
        self.length = torch.tensor(self.lengths, dtype=torch.float64, device=device)
        self.centers = torch.tensor(center/self.lengths, dtype=torch.float64, device=device)
        distance = torch.clamp(self.centers.square().sum(1)[:, None]+self.centers.square().sum(1)[None]-2*self.centers @ self.centers.T, min=0.)
        cc = torch.exp(-.5*distance)
        values, vectors = torch.linalg.eigh((cc+cc.T)/2)
        good = values > max(float(values.max())*1e-10, 1e-12)
        self.transform = vectors[:, good]/torch.sqrt(values[good])
        self.width = int(good.sum())+1

    @torch.no_grad()
    def __call__(self, w):
        z = (torch.tensor(np.asarray(w), dtype=torch.float64, device=self.device)-self.mean)/self.scale/self.length
        distance = torch.clamp(z.square().sum(1)[:, None]+self.centers.square().sum(1)[None]-2*z @ self.centers.T, min=0.)
        return torch.cat((torch.ones((len(z), 1), dtype=torch.float64, device=z.device), torch.exp(-.5*distance) @ self.transform), dim=1)


@torch.no_grad()
def statistics(data, dictionary, train_subjects, knots, chunk):
    device, width = dictionary.device, dictionary.width
    zeros = lambda shape: torch.zeros(shape, dtype=torch.float64, device=device)
    def empty():
        return dict(gram=zeros((width, width)),
            cross=[zeros((width, len(k))) for k in knots],
            response_gram=[zeros((len(k), len(k))) for k in knots],
            response_sum=[zeros((len(k),)) for k in knots],
            bin_counts=[np.zeros(len(k)-1, dtype=int) for k in knots], rows=0)
    all_stats, train_stats = empty(), empty()
    train_members = np.zeros(data.n, dtype=bool); train_members[train_subjects] = True
    for a in range(0, data.m, chunk):
        b = min(a+chunk, data.m); phi = dictionary(data.w[a:b])
        selected = train_members[data.sid[a:b]]
        which = torch.tensor(selected, device=device)
        all_stats['gram'] += phi.T @ phi
        train_stats['gram'] += phi[which].T @ phi[which]
        all_stats['rows'] += b-a; train_stats['rows'] += int(selected.sum())
        for c, k in enumerate(knots):
            h, bins = warp.hat_basis(data.y[a:b, c], k)
            ht = torch.tensor(h, dtype=torch.float64, device=device)
            for result, x, target, mask in ((all_stats, phi, ht, slice(None)),
                                           (train_stats, phi[which], ht[which], selected)):
                result['cross'][c] += x.T @ target
                result['response_gram'][c] += target.T @ target
                result['response_sum'][c] += target.sum(0)
                result['bin_counts'][c] += np.bincount(bins[mask], minlength=len(k)-1)
        if b == data.m or b//chunk % 32 == 0:
            print(f'[WarpSource] actual rows {b}/{data.m}', flush=True)
    def cpu(result):
        return {key: ([item.cpu().numpy() if torch.is_tensor(item) else item for item in value] if isinstance(value, list)
                       else value.cpu().numpy() if torch.is_tensor(value) else value) for key, value in result.items()}
    return cpu(all_stats), cpu(train_stats)


def create_model(stats, channel, knots, learn_warp):
    return warp.Model(stats['gram'], stats['cross'][channel], stats['response_gram'][channel],
        stats['response_sum'][channel], stats['bin_counts'][channel], knots, learn_warp=learn_warp)


def warm_start(target, source):
    with torch.no_grad():
        for name, value in target.named_parameters():
            value.copy_(dict(source.named_parameters())[name])


def analyze(source, output, centers=256, knot_count=33, steps=200, chunk=8192, seed=438, device='cuda'):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain earlier global-warp source: {output}')
    data = wave.io.MatrixCases(source, ('logit_nn', 'p_nn'), seed)
    data.coordinates(data.paths[0])
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed); train = np.sort(rng.permutation(data.n)[:int(.75*data.n)])
    train_mask = np.isin(data.sid, train)
    scaler = wave.sur.Scaler(data.w, data.d, row_mask=train_mask)
    subject_ids = rng.choice(train, min(centers, len(train)), replace=False)
    center_rows = np.array([rng.integers(data.offsets[s], data.offsets[s+1]) for s in subject_ids])
    dictionary = Dictionary(scaler.mean, scaler.scale, data.w[center_rows], data.d, device)
    knots = [np.unique(np.quantile(data.y[:, c], np.linspace(0., 1., knot_count))) for c in range(len(data.channels))]
    if min(map(len, knots)) < 3:
        raise ValueError('Global warp needs a nonconstant response distribution')
    protocol = dict(status='RUNNING', model=data.index['model'], source=str(Path(source).resolve()),
        all_subjects=data.n, all_original_rows=data.m, centers=len(center_rows), features=dictionary.width,
        seed=seed, maximum_knots=knot_count, steps_per_fit=steps, chunk=chunk, device=device, precision='float64',
        kernel='joint-W RBF, one U-block and one X-block median-distance length; constant feature',
        U_rotation_equivariant=True, link='positive-slope piecewise linear global map per Y channel',
        jacobian_included=True, knot_locations_use_all_original_Y=True,
        validation_interpretation='heldout original subject rows conditional on frozen knot locations',
        prediction_quality_policy='identity and learned warp both retained; continuous errors, no veto',
        code_sha256=hashlib.sha256(b''.join((wave.HERE/f).read_bytes() for f in
            ('joint_warp_source.py', 'warped_gp.py', 'matrix_io.py', 'surrogates.py'))).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    all_stats, training = statistics(data, dictionary, train, knots, chunk)
    if all_stats['rows'] != data.m:
        raise ArithmeticError('Warp source did not use all actual rows')
    np.savez_compressed(output/'dictionary.npz', mean=scaler.mean, scale=scaler.scale,
        centers_raw=np.asarray(data.w[center_rows]), center_rows=center_rows, block_lengths=dictionary.lengths,
        nystrom_transform=dictionary.transform.cpu().numpy(), training_subjects=train)
    stats_saved = dict(all_gram=all_stats['gram'], training_gram=training['gram'])
    for c in range(len(data.channels)):
        for label, value in (('all', all_stats), ('training', training)):
            for key in ('cross', 'response_gram', 'response_sum', 'bin_counts'):
                stats_saved[f'{label}_channel{c}_{key}'] = value[key][c]
    np.savez_compressed(output/'sufficient_statistics.npz', **stats_saved)
    models, histories = {}, []
    for c, channel in enumerate(data.channels):
        identity = create_model(training, c, knots[c], False); hi = warp.fit(identity, steps)
        learned = create_model(training, c, knots[c], True); warm_start(learned, identity)
        hw = warp.fit(learned, steps)
        models[c] = dict(identity=identity, learned=learned)
        histories.extend([dict(channel=channel, phase='identity_training', history=hi),
                          dict(channel=channel, phase='learned_training', history=hw)])
    # Evaluate actual original heldout rows; no interpolation or response formula.
    errors = {(c, kind): 0. for c in models for kind in ('identity', 'learned')}
    validation_rows = 0; prediction = {}
    for c, variants in models.items():
        for kind, model in variants.items():
            prediction[c, kind] = (model.coefficients().numpy(), model.heights().detach().numpy())
    for a in range(0, data.m, chunk):
        b = min(a+chunk, data.m); selected = ~train_mask[a:b]
        if not np.any(selected):
            continue
        phi = dictionary(np.asarray(data.w[a:b])[selected]).cpu().numpy()
        validation_rows += int(selected.sum())
        for (c, kind), (coef, heights) in prediction.items():
            raw_prediction = warp.inverse(phi @ coef, knots[c], heights)
            errors[c, kind] += float(np.sum((raw_prediction-data.y[a:b, c][selected])**2))
    quality, saved = [], {}
    for c, channel in enumerate(data.channels):
        variance = max(float(np.var(data.y[train_mask, c])), 1e-30)
        quality.append(dict(channel=channel, actual_validation_rows=validation_rows,
            identity_original_Y_validation_nmse=errors[c, 'identity']/validation_rows/variance,
            learned_original_Y_validation_nmse=errors[c, 'learned']/validation_rows/variance))
        saved[f'channel{c}_knots'] = knots[c]
        for kind in ('identity', 'learned'):
            full = create_model(all_stats, c, knots[c], kind == 'learned')
            warm_start(full, models[c][kind]); history = warp.fit(full, steps)
            histories.append(dict(channel=channel, phase=kind+'_all_subject_refit', history=history))
            saved[f'channel{c}_{kind}_heights'] = full.heights().detach().numpy()
            saved[f'channel{c}_{kind}_coefficients'] = full.coefficients().numpy()
            saved[f'channel{c}_{kind}_log_slopes'] = full.log_slopes.detach().numpy()
            print(f'[WarpSource] {data.index["model"]} {channel} {kind}: all {data.n} subjects refitted', flush=True)
    np.savez_compressed(output/'global_warps.npz', **saved)
    wave.csv_file(output/'original_row_validation.csv', quality)
    wave.json_file(output/'optimization_history.json', histories)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(data.read_files)), forbidden_files_read=[],
        all_actual_rows_in_sufficient_statistics=all_stats['rows'], all_subjects=data.n))
    files = ('dictionary.npz', 'global_warps.npz', 'original_row_validation.csv', 'optimization_history.json')
    hashes = {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in files}
    wave.json_file(output/'manifest.json', hashes)
    protocol.update(status='COMPLETE', manifest_sha256=hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest())
    wave.json_file(output/'protocol.json', protocol)
    print('[WarpSource] Original-row validation:', quality, flush=True)
    return quality


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--run-name', default='waveBE_global_warped_GP_source_seed438')
    args = p.parse_args(); source = wave.io.default_step2(wave.ROOT, args.model)
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        analyze(source, source.parent/'step5_matrix_metrics'/args.run_name)
