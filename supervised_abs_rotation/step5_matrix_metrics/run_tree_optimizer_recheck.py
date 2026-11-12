"""Recheck frozen blind tree optima in exactly the paired AW subject draws."""
import hashlib
import json

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import tree_rotation_stability as stability


def main():
    source = wave.io.default_step2(wave.ROOT, 'CR3'); parent = source.parent/'step5_matrix_metrics'
    original = parent/'waveAU_conditional_UX_tree_complexity_seed421'
    earlier = parent/'waveAW_tree_blind_bootstrap_seed425'
    output = parent/'waveAW2_frozen_optimizer_paired_recheck'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous optimizer recheck: {output}')
    protocol = json.loads((earlier/'protocol.json').read_text())
    frozen = json.loads((earlier/'blind_freeze.json').read_text())
    if protocol['status'] != 'COMPLETE':
        raise ValueError('Blind optimization must be complete before a follow-up')
    for filename, digest in frozen.items():
        if hashlib.sha256((earlier/filename).read_bytes()).hexdigest() != digest:
            raise ValueError('Earlier frozen blind evidence changed')
    output.mkdir(parents=True)
    wave.json_file(output/'protocol.json', dict(status='RUNNING', source_wave=str(earlier.resolve()),
        optimized_rotations_refitted=False, original_subject_draws_replayed=True,
        no_prediction_MSE_veto=True, n_subjects=protocol['n_subjects'],
        source_blind_file_sha256=frozen, code_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest()))
    bank = profiles.MatrixBank(source, parent/'matrix_response_table_seed383'); data = bank.data
    operators = [np.load(original/f'X_contrast_column{j}.npy', allow_pickle=False) for j in range(data.dx)]
    targets, residuals = stability.coupling_modes(bank.responses, operators)
    z = data.u-data.u.mean(0); z /= np.sqrt(np.var(data.u, axis=0).mean())
    with np.load(earlier/'blind_rotation_matrices.npz', allow_pickle=False) as saved:
        random_rotations, optimized = saved['random'], saved['optimized']
    rng = np.random.default_rng(protocol['seed'])
    replay = np.array([stability.random_rotation(data.d, rng) for _ in range(protocol['random_SO_rotations'])])
    if not np.array_equal(replay, random_rotations):
        raise ArithmeticError('Replay failed to reproduce the earlier RNG state')
    losses = []
    for b in range(protocol['paired_subject_bootstrap_refits']):
        ids = rng.integers(data.n, size=data.n)
        sampled, discarded = stability.sample_functions(targets, residuals, ids)
        losses.append([stability.score(z[ids], q, sampled, discarded)[0] for q in optimized])
        np.savez_compressed(output/'blind_optimizer_bootstrap_scores.npz', losses=np.asarray(losses))
        print(f'[WaveAW2] paired frozen-optimizer refits {b+1}/{protocol["paired_subject_bootstrap_refits"]}', flush=True)
    digest = hashlib.sha256((output/'blind_optimizer_bootstrap_scores.npz').read_bytes()).hexdigest()
    wave.json_file(output/'blind_freeze.json', dict(scores_sha256=digest))
    # Evaluation labels are read only after the fixed-rotation scores freeze.
    reveal = json.loads((original/'reveal.json').read_text())['mapping']
    base_case = next(int(k.split('_')[1]) for k, value in reveal.items() if value == 'baseline')
    with np.load(original/'geometry.npz', allow_pickle=False) as geometry:
        group = int(geometry['membership'][base_case])
    with np.load(earlier/'blind_bootstrap_scores.npz', allow_pickle=False) as saved:
        baseline = saved['losses'][:, group]
    comparison = []
    for k in range(len(optimized)):
        margin = (np.asarray(losses)[:, k]-baseline)/baseline
        comparison.append(dict(optimizer=k, relative_margin_q05=float(np.quantile(margin, .05)),
            relative_margin_median=float(np.median(margin)), relative_margin_q95=float(np.quantile(margin, .95)),
            fraction_strictly_below_baseline=float(np.mean(margin < 0.))))
    wave.csv_file(output/'paired_comparisons.csv', comparison)
    wave.json_file(output/'assessment.json', dict(optimized_rotations=comparison,
        interpretation='exploratory paired subject refit stability of the already frozen optima; not independent new-subject validation'))
    wave.json_file(output/'protocol.json', dict(status='COMPLETE', source_wave=str(earlier.resolve()),
        original_subject_draws_replayed=True, optimized_rotations_refitted=False, score_values_sha256=digest,
        n_subjects=data.n, paired_refits=len(losses), no_prediction_MSE_veto=True))
    print('[WaveAW2]', comparison, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        main()
