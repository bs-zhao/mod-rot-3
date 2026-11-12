"""Transfer already frozen blind rotations to poly/RBF function sources."""
import hashlib
import json

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import fitted_bank
import ux_predictive_complexity as uc
import conditional_ux
import tree_rotation_stability as stability


def main():
    source = wave.io.default_step2(wave.ROOT, 'CR3'); parent = source.parent/'step5_matrix_metrics'
    earlier = parent/'waveAW_tree_blind_bootstrap_seed425'
    output = parent/'waveAY2_frozen_counterexample_transfer'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous source transfer: {output}')
    old = json.loads((earlier/'protocol.json').read_text())
    rotation_file = earlier/'blind_rotation_matrices.npz'
    expected = json.loads((earlier/'blind_freeze.json').read_text())[rotation_file.name]
    if old['status'] != 'COMPLETE' or hashlib.sha256(rotation_file.read_bytes()).hexdigest() != expected:
        raise ValueError('Earlier blind rotation search is incomplete or changed')
    with np.load(rotation_file, allow_pickle=False) as saved:
        rotations = saved['optimized']
    bank = fitted_bank.FittedBank(source, parent/'UX_complexity_source_seed383'); data = bank.data
    z = data.u-data.u.mean(0); z /= np.sqrt(np.var(data.u, axis=0).mean())
    output.mkdir(parents=True)
    configuration = dict(status='RUNNING', models=['poly5', 'rbf2'], all_subjects=data.n,
        optimizer_rotations_fitted_again=False, source_fitted_manifest_sha256=bank.source_digest,
        frozen_rotation_sha256=expected, prediction_error_policy='continuous diagnostic, no fixed veto',
        code_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest())
    wave.json_file(output/'protocol.json', configuration)
    blind, sources = [], {}
    for name in ('poly5', 'rbf2'):
        original = parent/f'waveAY_approximate_UX_tree_seed421_{name}'
        protocol = json.loads((original/'protocol.json').read_text())
        if protocol['status'] != 'COMPLETE' or protocol['fitted_source_manifest_sha256'] != bank.source_digest:
            raise ValueError('Approximate-source tree experiment is not frozen or has a different function bank')
        coefficients, zs, meta = uc.source_coefficients(bank, name)
        table = np.einsum('np,pcm->ncm', conditional_ux.source_features(zs, meta), coefficients, optimize=True)
        operators = [np.load(original/f'X_contrast_column{j}.npy', allow_pickle=False) for j in range(data.dx)]
        targets, residuals = stability.coupling_modes(table, operators)
        sampled, discarded = stability.sample_functions(targets, residuals, np.arange(data.n))
        for k, q in enumerate(rotations):
            loss, counts = stability.score(z, q, sampled, discarded)
            blind.append(dict(source_estimator=name, frozen_optimizer=k, loss=loss, per_X_counts=counts))
        sources[name] = original
    wave.csv_file(output/'blind_transfer_scores.csv', blind)
    digest = hashlib.sha256((output/'blind_transfer_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'blind_freeze.json', dict(scores_sha256=digest))
    results = []
    for name, original in sources.items():
        reveal = json.loads((original/'reveal.json').read_text())['mapping']
        case = next(int(k.split('_')[1]) for k, value in reveal.items() if value == 'baseline')
        with np.load(original/'geometry.npz', allow_pickle=False) as geometry:
            group = geometry['membership'][case]
        with np.load(original/'per_X_complexity.npz', allow_pickle=False) as saved:
            baseline = float(np.max(saved['counts'][group, 0, 2]))
        for row in blind:
            if row['source_estimator'] == name:
                results.append(dict(**row, baseline_loss=baseline, relative_margin=(row['loss']-baseline)/baseline))
    wave.csv_file(output/'transfer_assessment.csv', results)
    configuration.update(status='COMPLETE', scores_sha256=digest)
    wave.json_file(output/'protocol.json', configuration)
    print('[WaveAY2]', results, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
