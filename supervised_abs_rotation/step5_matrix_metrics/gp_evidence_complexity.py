"""BD: normalized GP expected description cost of each conditional X function.

Uses the already fitted GP marginal objective, rather than treating its
lengthscale parameters as effect magnitudes. Output tasks were normalized
before fitting. This is expected scalar-function log density over the X
feature ensemble, not an assertion of independent observed X profiles.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import run as wave


def analyze(source, original, output):
    original, output = Path(original), Path(output)
    old = json.loads((original/'protocol.json').read_text())
    original_digest = hashlib.sha256((original/'blind_scores.csv').read_bytes()).hexdigest()
    if old['status'] != 'COMPLETE' or old['scores_sha256'] != original_digest:
        raise ValueError('Source GP is incomplete or changed')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Retain previous GP evidence audit: {output}')
    output.mkdir(parents=True)
    with np.load(original/'GP_hyperparameters.npz', allow_pickle=False) as saved:
        cost = np.min(saved['evidence_loss'], axis=0)/old['n_subjects']
    with np.load(original/'data_geometry.npz', allow_pickle=False) as saved:
        membership = saved['case_membership']
    channels = old['channels']; dx = len(cost)//len(channels)
    cost = cost.reshape(len(channels), dx, -1)
    blind = []
    for k, group in enumerate(membership):
        for c, channel in enumerate(channels):
            for pooling, loss in (('Xmean', cost[c, :, group].mean()), ('Xworst', cost[c, :, group].max())):
                blind.append(dict(anonymous_case=f'case_{k:03d}', estimator='GP_function_evidence', channel=channel,
                    metric='conditional_UX_GP_expected_NLL_'+pooling, loss=float(loss)))
    wave.csv_file(output/'blind_scores.csv', blind)
    digest = hashlib.sha256((output/'blind_scores.csv').read_bytes()).hexdigest()
    wave.json_file(output/'protocol.json', dict(status='FROZEN', scores_sha256=digest,
        source_GP_scores_sha256=original_digest, all_subjects=old['n_subjects'],
        target_preprocessing=old.get('global_output_transform', 'identity'),
        interpretation=__doc__, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        prediction_quality_policy='continuous diagnostic, no fixed veto'))
    reveal = json.loads((original/'reveal.json').read_text())
    if reveal['scores_sha256'] != original_digest:
        raise ValueError('Original GP label mapping changed')
    wave.json_file(output/'reveal.json', dict(scores_sha256=digest, mapping=reveal['mapping']))
    index = json.loads((Path(source)/'index.json').read_text())
    grouped = defaultdict(dict)
    for row in blind:
        grouped[row['channel'], row['metric']][reveal['mapping'][row['anonymous_case']]] = row['loss']
    rows = [dict(estimator='GP_function_evidence', channel=c, metric=m,
        **wave.st.evaluate_curve(v, index['cases']), is_control=False, prediction_error_veto_applied=False)
        for (c, m), v in grouped.items()]
    wave.csv_file(output/'metric_summary.csv', rows)
    protocol = json.loads((output/'protocol.json').read_text()); protocol['status'] = 'COMPLETE'
    wave.json_file(output/'protocol.json', protocol)
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3'); p.add_argument('--available', action='store_true')
    p.add_argument('--source-run', default='waveAZ_normal_rank_conditional_UX_GP_seed416')
    p.add_argument('--run-name', default='waveBD_ordinal_GP_evidence')
    args = p.parse_args()
    for model in (('CR3', 'C1', 'C2', 'C3') if args.available else (args.model,)):
        source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
        original = parent/args.source_run
        if not (original/'protocol.json').exists() or json.loads((original/'protocol.json').read_text())['status']!='COMPLETE':
            print(f'[WaveBD] {model}: GP source still running or absent', flush=True)
            continue
        rows = analyze(source, original, parent/args.run_name)
        print(f'[WaveBD] {model}:', [(r['channel'], r['metric'], r['success_planes'], r['baseline_rank']) for r in rows], flush=True)
