"""AZ: GP UX relevance after a single global ordinal response calibration."""
import argparse

import torch
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as raw
import rank_response_profiles as ranked
import run_matrix_gp_ard as gp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=wave.io.DIMS, default='CR3')
    p.add_argument('--transform', choices=('normal_rank', 'uniform_rank'), default='normal_rank')
    args = p.parse_args()
    source = wave.io.default_step2(wave.ROOT, args.model); parent = source.parent/'step5_matrix_metrics'
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        raw_folder = parent/'matrix_response_table_seed383'
        if not raw_folder.exists():
            raw.prepare(source, raw_folder)
        folder = parent/f'matrix_response_{args.transform}_seed383'
        if not folder.exists():
            ranked.prepare(source, folder, raw_folder, transform=args.transform)
        rows = gp.analyze(source, parent/f'waveAZ_{args.transform}_conditional_UX_GP_seed416', folder,
            X_contrast='conditional_witness', bank_factory=ranked.MatrixBank)
        print('[WaveAZ] Primary grid candidates, without prediction-MSE veto:',
              sum(row['all_planes_pass'] and not row['is_control'] for row in rows), flush=True)


if __name__ == '__main__':
    main()
