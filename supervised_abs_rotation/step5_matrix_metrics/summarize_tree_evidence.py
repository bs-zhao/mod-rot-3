"""Report frozen four-model tree evidence continuously, without an MSE veto."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import run as wave


def summarize(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metric = 'UX_function_tree_leaves_relative_L2_eps100_Xworst'
    summary, full_curves, provenance = [], [], []
    figure, axes = plt.subplots(2, 2, figsize=(17, 10), constrained_layout=True)
    for model, axis in zip(('CR3', 'C1', 'C2', 'C3'), axes.ravel()):
        source = wave.io.default_step2(wave.ROOT, model)
        folder = source.parent/'step5_matrix_metrics/waveAU_conditional_UX_tree_complexity_seed421'
        protocol = json.loads((folder/'protocol.json').read_text())
        digest = hashlib.sha256((folder/'blind_scores.csv').read_bytes()).hexdigest()
        reveal = json.loads((folder/'reveal.json').read_text())
        if protocol['status'] != 'COMPLETE' or digest != protocol['scores_sha256'] or digest != reveal['scores_sha256']:
            raise ValueError(f'{model} scores are incomplete or changed')
        with (folder/'blind_scores.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        losses = {reveal['mapping'][r['anonymous_case']]: float(r['loss']) for r in rows
                  if r['channel'] == 'logit_nn' and r['metric'] == metric}
        with (folder/'fit_quality.csv').open() as stream:
            quality = next(r for r in csv.DictReader(stream) if r['channel'] == 'logit_nn')
        index = json.loads((source/'index.json').read_text())
        evaluation = wave.st.evaluate_curve(losses, index['cases'])
        baseline = losses['baseline']; minimum = min(losses.values()); pairs = {}
        for case in index['cases']:
            if case.get('pair_1based'):
                pair = tuple(case['pair_1based'])
                pairs.setdefault(pair, {})[case['angle_deg']] = losses[case['id']]
                full_curves.append(dict(model=model, pair=f'{pair[0]}-{pair[1]}', angle=case['angle_deg'],
                    leaf_count=losses[case['id']], baseline_leaf_count=baseline,
                    relative_loss=losses[case['id']]/baseline))
        for color, (pair, values) in enumerate(sorted(pairs.items())):
            angles = [0]+sorted(values)
            axis.plot(angles, [1.]+[values[a]/baseline for a in sorted(values)], marker='.',
                      linestyle='-' if color < 10 else '--', color=plt.get_cmap('tab20')(color),
                      linewidth=1., markersize=4, label=f'U{pair[0]}-U{pair[1]}')
        axis.axhline(1., color='black', linewidth=.8, linestyle='--')
        axis.set_title(f'{model}: {evaluation["success_planes"]}/{evaluation["n_planes"]} strict planes; '
                       f'baseline rank {evaluation["baseline_rank"]}')
        axis.set_xlabel('Rotation angle (degrees)'); axis.set_ylabel('Rule count / baseline rule count')
        axis.set_xticks([0, 15, 30, 45, 60, 75, 90]); axis.grid(alpha=.2)
        axis.legend(fontsize=7, ncol=1, loc='upper left', bbox_to_anchor=(1.01, 1.), frameon=False)
        summary.append(dict(model=model, channel='logit_nn', metric=metric, **evaluation,
            baseline_leaf_count=baseline, minimum_leaf_count=minimum,
            best_observed_relative_advantage_over_baseline=(baseline-minimum)/baseline,
            source_profile_validation_nmse=float(quality['validation_nmse']),
            worst_full_tree_heldout_relative_mse=float(quality['worst_full_tree_subject_validation_relative_mse']),
            prediction_error_policy='continuous diagnostic, no fixed veto', validation_status='exploratory grid evidence'))
        provenance.append(dict(model=model, source_run=str(folder.resolve()), scores_sha256=digest,
                               definition_unchanged=True, all_subjects=protocol['n_subjects']))
    figure.suptitle('One frozen conditional UX function tree criterion across four matrices\n'
                   'Worst-X leaves for 10% relative L2 approximation; every plane shown', fontsize=13)
    figure.savefig(output/'rotation_curves.png', dpi=160); plt.close(figure)
    wave.csv_file(output/'four_model_evidence.csv', summary)
    wave.csv_file(output/'all_plane_curves.csv', full_curves)
    wave.json_file(output/'provenance.json', provenance)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=wave.ROOT/'outputs/supervised_abs_rotation/waveAU_four_model_evidence')
    args = p.parse_args()
    for row in summarize(args.output):
        print(row)
