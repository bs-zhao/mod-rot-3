"""Read and verify frozen independent-Haar results; draw their empirical CDFs."""
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import run as wave
from random_rotation_percentile import summarize


def main():
    root=wave.ROOT/'outputs/supervised_abs_rotation/waveCL_independent_Haar_percentile_seed541'
    plan_file=root/'plan_frozen_before_sampling.json';plan=json.loads(plan_file.read_text())
    plan_hash=hashlib.sha256(plan_file.read_bytes()).hexdigest();rows=[]
    fig,axes=plt.subplots(1,4,figsize=(14,3.6),constrained_layout=True)
    for ax,model in zip(axes,plan['models']):
        folder=root/model;protocol=json.loads((folder/'protocol.json').read_text())
        if protocol['status']!='COMPLETE' or protocol['plan_sha256']!=plan_hash:raise ValueError('Independent model incomplete or plan changed')
        for name,key in (('random_scores.csv','scores_sha256'),('random_frames.npz','frames_sha256')):
            if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=protocol[key]:raise ValueError('Frozen independent scores/frames changed')
        saved=json.loads((folder/'assessment.json').read_text())
        with (folder/'random_scores.csv').open() as stream:losses=np.array([float(r['loss']) for r in csv.DictReader(stream)])
        result=summarize(losses,saved['reference_loss'],family_size=plan['confidence_family'],alpha=plan['simultaneous_alpha'],
            relative_numerical_tolerance=plan['numerical_relative_tie_tolerance'])
        for key in ('random_at_least_as_good_as_reference','random_numerical_ties','random_favorable_tail_upper_percent'):
            if not np.isclose(result[key],saved[key],rtol=1e-12,atol=1e-14):raise ArithmeticError('Saved percentile assessment did not replay')
        row=dict(model=model,channel=plan['channel'],all_subjects=saved['all_subjects'],**result);rows.append(row)
        relative=np.sort(100*(losses-saved['reference_loss'])/abs(saved['reference_loss']))
        ax.step(np.r_[0.,relative],np.r_[0.,100*np.arange(1,len(relative)+1)/len(relative)],where='post',lw=1.5)
        ax.scatter(relative,100*np.arange(1,len(relative)+1)/len(relative),s=9,alpha=.6)
        ax.axvline(0,color='black',lw=.8,ls='--');ax.axhline(15,color='gray',ls=':',lw=.8)
        ax.errorbar([0],[0],yerr=[[0],[result['random_favorable_tail_upper_percent']]],color='tab:orange',capsize=4,lw=1.5)
        ax.set_ylim(-3,103);ax.grid(alpha=.15)
        ax.set_title(f'{model}: {result["random_at_least_as_good_as_reference"]}/{len(losses)} no worse\n95% simultaneous upper {result["random_favorable_tail_upper_percent"]:.2f}%',fontsize=10)
        ax.set_xlabel('Random score change from reference (%)')
    axes[0].set_ylabel('Empirical Haar-score CDF (%)')
    fig.suptitle('Fixed NN-p Shapley UX mode support: 64 new Haar rotations per model, N=5000\n'
                 'Dashed vertical: reference; orange: its one-sided simultaneous confidence bound',fontsize=11)
    fig.savefig(root/'independent_Haar_score_CDFs.png',dpi=160);plt.close(fig)
    wave.csv_file(root/'verified_percentile_summary.csv',rows)
    wave.json_file(root/'report_verification.json',dict(status='COMPLETE',plan_sha256=plan_hash,
        completed_models=[r['model'] for r in rows],all_NN_probability_responses=True,
        global_minimum_or_unique_axes_proven=False,random_CDF_and_confidence_assessments_replayed=True))
    print('[Verified independent Haar percentile figure]',root/'independent_Haar_score_CDFs.png',flush=True)


if __name__=='__main__':main()
