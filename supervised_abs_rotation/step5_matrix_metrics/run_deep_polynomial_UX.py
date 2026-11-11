"""Wave AA: higher-degree joint regression, then all-trial UX support."""
import argparse
from threadpoolctl import threadpool_limits

import run as wave
import joint_bank
import run_all_trial_support as support


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--degree', type=int, default=5)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--run-name', default='waveAA_poly5_UX_all_trials')
    args = p.parse_args()
    if args.degree < 2:
        p.error('Degree >=2 required')
    survivors = None
    with threadpool_limits(limits=4):
        for model in (('CR3', 'C1', 'C2', 'C3') if args.batch else (args.model,)):
            source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
            bank = parent/f'joint_bank_degree{args.degree}_seed391'
            if not bank.exists():
                joint_bank.prepare(source, bank, degrees=(args.degree,), device='cuda')
            rows = support.analyze(source, parent/args.run_name, bank)
            passed = {(r['estimator'], r['channel'], r['metric']) for r in rows
                      if r['adequate'] and r['all_planes_pass'] and r['metric'].startswith('UX_')}
            survivors = passed if survivors is None else survivors & passed
            print(f'[WaveAA] {model}: {len(passed)} primary UX sparsity passes; {len(survivors)} common criteria', flush=True)
            if args.batch and not survivors:
                print('[WaveAA] No identical UX criterion remains; later datasets skipped', flush=True)
                return


if __name__ == '__main__':
    main()
