"""Transfer previously frozen blind frames without optimizing the new criterion."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import learned_response_profiles as learned
from conditional_kernel_blind import Scorer


def analyze(source, original, bank_folder, counterexample, output, learned_link=False):
    original, counterexample, output = map(Path, (original, counterexample, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain earlier counterexample transfer: {output}')
    protocol = json.loads((original/'protocol.json').read_text())
    if protocol['status'] != 'COMPLETE':
        raise ValueError('Conditional moment source is incomplete')
    source_digest = hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()
    if source_digest != protocol['scores_sha256']:
        raise ValueError('Conditional moment source scores changed')
    frozen = json.loads((counterexample/'blind_freeze.json').read_text())
    rotations_digest = hashlib.sha256((counterexample/'blind_rotations.npz').read_bytes()).hexdigest()
    if rotations_digest != frozen['rotations_sha256']:
        raise ValueError('Frozen blind rotations changed')
    bank = (learned.MatrixBank if learned_link else profiles.MatrixBank)(source, bank_folder)
    if bank.protocol['manifest_sha256'] != protocol['source_matrix_manifest_sha256']:
        raise ValueError('Matrix response source changed')
    output.mkdir(parents=True)
    scorer = Scorer(bank, original)
    v, expected = scorer.exact_cache[0]
    reproduced = scorer.direction(v, force=True)
    if not np.allclose(expected, reproduced, rtol=1e-7, atol=1e-10):
        raise ArithmeticError('Transfer scorer did not reproduce its frozen source statistic')
    with np.load(counterexample/'blind_rotations.npz', allow_pickle=False) as saved:
        q = saved['optimized'].copy()
    score = scorer.score(q)
    wave.csv_file(output/'frozen_transfer_scores.csv', [dict(frame='previously_optimized', loss=score)])
    digest = hashlib.sha256((output/'frozen_transfer_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'blind_freeze.json', dict(scores_sha256=digest, rotations_sha256=rotations_digest))
    # Evaluation labels enter only after the new score has been frozen.
    reveal = json.loads((original/'reveal.json').read_text())['mapping']
    k = next(int(k.split('_')[1]) for k,v in reveal.items() if v=='baseline')
    with np.load(original/'W_only_geometry.npz', allow_pickle=False) as saved:
        qbase = saved['directions'][saved['membership'][k]].T
    baseline = scorer.score(qbase)
    assessment = dict(baseline_loss=baseline, transferred_frame_loss=score,
        relative_margin=(score-baseline)/baseline, all_subjects=bank.data.n,
        original_rows=bank.data.m, new_optimization_performed=False,
        Ui_kernel_centers=scorer.centers, source_scores_sha256=source_digest,
        source_response_transform=protocol.get('global_output_transform', 'identity'))
    wave.json_file(output/'assessment.json', assessment)
    wave.json_file(output/'new_U_feature_audits.json', scorer.audits)
    wave.json_file(output/'read_audit.json', dict(files=sorted(set(bank.data.read_files)), forbidden_files_read=[]))
    wave.json_file(output/'protocol.json', dict(status='COMPLETE', **assessment, scores_sha256=digest))
    print('[Frozen kernel transfer]', assessment, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--learned-link', action='store_true')
    args = p.parse_args()
    with threadpool_limits(limits=4):
        source = wave.io.default_step2(wave.ROOT, 'CR3'); parent = source.parent/'step5_matrix_metrics'
        tag = 'BF' if args.learned_link else 'BC2'
        original = parent/('waveBF_learned_global_link_kernel_seed430' if args.learned_link else
                           'waveBC2_conditional_kernel_centers32_seed430')
        bank_folder = parent/('matrix_response_global_warp_learned_seed438' if args.learned_link else
                              'matrix_response_table_seed383')
        analyze(source, original, bank_folder, parent/'waveBC3_conditional_kernel_blind_seed439',
                parent/f'wave{tag}_frozen_counter_transfer', args.learned_link)
