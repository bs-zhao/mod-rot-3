"""Evaluate frozen blind frames' axis positions after sign/permutation matching."""
import hashlib
import json

import numpy as np
from scipy.optimize import linear_sum_assignment

import run as wave
import matrix_function_profiles as profiles


def axis_angles(reference,frame):
    cosine=np.abs(reference.T@frame)
    rows,columns=linear_sum_assignment(-cosine)
    values=np.degrees(np.arccos(np.clip(cosine[rows,columns],0.,1.)))
    return dict(matched_mean_angle_deg=float(values.mean()),matched_maximum_angle_deg=float(values.max()),
        matched_angles_deg=values.tolist(),matched_frame_columns=columns.tolist(),
        assignment='maximize sum of absolute axis inner products; signs/permutations equivalent')


def main():
    output=wave.ROOT/'outputs/supervised_abs_rotation/frozen_blind_axis_positions'
    output.mkdir(parents=True,exist_ok=True);rows=[];audits=[]
    sources=[('BB','waveBB_radial_GP_fixed_function_seed416',('CR3','C1','C2','C3')),
        ('BB2','waveBB2_rank_radial_fixed_function_seed416',('CR3','C1','C2','C3')),
        ('BC3','waveBC3_conditional_kernel_blind_seed439',('CR3',)),
        ('BK2','waveBK2_complete_response_blind_seed456',('CR3',))]
    for tag,name,models in sources:
        for model in models:
            source=wave.io.default_step2(wave.ROOT,model);parent=source.parent/'step5_matrix_metrics';folder=parent/name
            if not (folder/'assessment.json').exists():continue
            protocol=json.loads((folder/'protocol.json').read_text())
            if protocol['status']!='COMPLETE':continue
            frozen=json.loads((folder/'blind_freeze.json').read_text())
            digest=hashlib.sha256((folder/'blind_rotations.npz').read_bytes()).hexdigest()
            if digest!=frozen.get('rotations_sha256',frozen.get('rotation_scores_sha256')):
                raise ValueError('A frozen optimized frame changed')
            bank=profiles.MatrixBank(source,parent/'matrix_response_table_seed383');data=bank.data
            baseline=next(path for path in data.paths if path.parent.name=='baseline')
            reference,_=wave.io.registration(data.u,data.coordinates(baseline))
            with np.load(folder/'blind_rotations.npz',allow_pickle=False) as saved:
                optimized=saved['optimized'].copy()
                index=int(np.argmin(saved['optimized_loss'])) if optimized.ndim==3 else None
                frame=optimized[index] if index is not None else optimized
            assessment=json.loads((folder/'assessment.json').read_text())
            loss=assessment.get('blind_optimized_minimum_loss',assessment.get('exact_blind_minimum_loss'))
            base=assessment['baseline_loss']
            rows.append(dict(wave=tag,model=model,optimized_loss=loss,baseline_loss=base,
                relative_change_percent=100*(loss-base)/max(abs(base),1e-30),
                **axis_angles(reference,frame),frozen_frame_sha256=digest))
            audits.append(dict(wave=tag,model=model,files=sorted(set(data.read_files)),forbidden_files_read=[]))
    wave.json_file(output/'axis_positions.json',rows)
    wave.csv_file(output/'axis_positions.csv',[{k:(json.dumps(v) if isinstance(v,list) else v) for k,v in row.items()} for row in rows])
    wave.json_file(output/'read_audit.json',audits)
    print('[Frozen blind positions]',[(r['wave'],r['model'],round(r['matched_maximum_angle_deg'],2),round(r['relative_change_percent'],4)) for r in rows])


if __name__=='__main__':main()
