"""Publication-style checkpoint for one uniformly defined confirmed single metric."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent/'results'


def main():
    metric='hs_b02_unique_edges_soft_t0.02';label='b02_three_gap_batch';models=['C1','C2','C3','CR3'];rows=[]
    for model in models:
        path=ROOT/label/model/'gaussian256'/'starts64'/(metric+'.json');j=json.loads(path.read_text(encoding='utf8'))
        refined=json.loads((ROOT/'plane_refinements'/label/model/(metric+'_gaussian256_starts64.json')).read_text(encoding='utf8'))
        g=j['gaps'];ev=g['G3']['evaluation'];oracle=g['G3']['oracle']
        rows.append(dict(model=model,g1_angle=g['G1']['best']['axis_error'],g2_angle=g['G2']['best']['axis_error'],
            g1_refined=refined['results']['G1']['refined_plane_gate'],g2_refined=refined['results']['G2']['refined_plane_gate'],
            mean=ev['matched_mean'],oracle_mean=oracle['matched_mean'],offmax=ev['off_max'],allowed_offmax=oracle['off_max']+.02))
    folder=ROOT/'batch_search';folder.mkdir(exist_ok=True)
    with (folder/'confirmed_single.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.1));fig.subplots_adjust(left=.055,right=.985,bottom=.18,top=.78,wspace=.3)
    x=np.arange(4);blue='#34789b';orange='#d68b3a'
    axes[0].bar(x-.17,[r['g1_angle'] for r in rows],.32,label='G1 analytic',color=blue)
    axes[0].bar(x+.17,[r['g2_angle'] for r in rows],.32,label='G2 inferred',color=orange)
    axes[0].axhline(6,color='#555',ls='--',lw=1);axes[0].set_ylim(0,6.8);axes[0].set_ylabel('Worst recovered-axis error (degrees)')
    axes[0].set_title('G1 / G2: pass');axes[0].legend(frameon=False,fontsize=8)
    axes[1].bar(x-.17,[r['oracle_mean'] for r in rows],.32,label='Supervised reference',color='#bfc7cd')
    axes[1].bar(x+.17,[r['mean'] for r in rows],.32,label='Hessian objective',color=blue)
    for i,r in enumerate(rows):axes[1].plot([i-.4,i+.4],[r['oracle_mean']-.01]*2,color='#333',lw=1.3)
    axes[1].set_ylim(.6,1.01);axes[1].set_ylabel('TEST mean matched correlation');axes[1].set_title('G3: mean correspondence');axes[1].legend(frameon=False,fontsize=8)
    axes[2].bar(x,[r['offmax'] for r in rows],.5,color=blue)
    for i,r in enumerate(rows):axes[2].plot([i-.4,i+.4],[r['allowed_offmax']]*2,color='#333',lw=1.3)
    axes[2].set_ylim(0,max(.75,max(r['offmax'] for r in rows)*1.1));axes[2].set_ylabel('Largest off-target correlation');axes[2].set_title('G3: off-target mixing')
    for ax in axes:
        ax.set_xticks(x,models);ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
    fig.suptitle('One single objective across all four models: unique Hessian edges, soft-count t=0.02',fontsize=12,y=.96)
    fig.text(.055,.045,'64 starts; continuous plane verification. Black G3 marks are fixed acceptance limits. G3 has not passed.',fontsize=9)
    fig.savefig(folder/'confirmed_single_gaps.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
