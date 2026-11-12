"""BB2: fixed radial-source audit of the ordinal posterior UX candidate."""
import json

import torch
from threadpoolctl import threadpool_limits

import run as wave
import matrix_function_profiles as raw
import rank_response_profiles as ranked
import run_isotropic_gp_rotation as radial


def main():
    for model in ('CR3', 'C1', 'C2', 'C3'):
        source = wave.io.default_step2(wave.ROOT, model); parent = source.parent/'step5_matrix_metrics'
        raw_folder = parent/'matrix_response_table_seed383'
        if not raw_folder.exists():
            raw.prepare(source, raw_folder)
        shared = parent/'matrix_response_normal_rank_seed383'
        ready = (shared/'protocol.json').exists() and json.loads((shared/'protocol.json').read_text())['status']=='COMPLETE'
        # The other continuation can prepare its own cache concurrently.
        # Use a separate cache if that shared preparation is not complete.
        folder = shared if ready else parent/'matrix_response_normal_rank_radial_seed383'
        if not folder.exists():
            ranked.prepare(source, folder, raw_folder)
        rows = radial.analyze(source, parent/'waveBB2_rank_radial_fixed_function_seed416', folder,
            bank_factory=ranked.MatrixBank, primary_metric='pooled_UX_columns_worst_soft_count_eps003')
        print(f'[WaveBB2] {model}: {sum(r["all_planes_pass"] and not r["is_control"] for r in rows)} '
              'primary grid combinations; no prediction-MSE veto', flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        torch.set_num_threads(4)
        main()
