"""AY: polynomial/RBF approximation first, then conditional UX tree complexity."""
import argparse
import json

from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as profiles
import prepare_response_bank
import run_tree_function_complexity as trees


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3'); p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveAY_approximate_UX_tree_seed421')
    args = p.parse_args(); all_rows = []
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            matrix = parent/'matrix_response_table_seed383'
            if not matrix.exists():
                profiles.prepare(source, matrix)
            fitted = parent/'UX_complexity_source_seed383'
            if fitted.exists() and not {'poly5', 'rbf2'} <= set(json.loads((fitted/'protocol.json').read_text())['estimators']):
                fitted = parent/'UX_piece_source_seed383'
            if not fitted.exists():
                prepare_response_bank.prepare(source, fitted, estimators=('poly5', 'rbf2'))
            for name in ('poly5', 'rbf2'):
                rows = trees.analyze(source, parent/(args.run_name+'_'+name), matrix,
                                     source_estimator=name, fitted_folder=fitted)
                all_rows.extend(dict(model=model, **row) for row in rows)
                print(f'[WaveAY] {model} {name}: {sum(r["all_planes_pass"] for r in rows)} grid passes; '
                      'prediction error reported without a fixed veto', flush=True)
            # This is a four-model source-resolution audit of an existing
            # family; small grid deviations do not stop the remaining models.
    wave.csv_file(wave.ROOT/'outputs/supervised_abs_rotation'/f'{args.run_name}_all_models.csv', all_rows)


if __name__ == '__main__':
    main()
