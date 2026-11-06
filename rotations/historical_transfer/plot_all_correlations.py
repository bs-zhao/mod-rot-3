"""One complete figure: every method x four models x both representations.

Uses persisted signed TEST correlations, never refits or rematches on TEST.
Rows inside each panel are original u axes; columns are true parameters.
Black cell outlines show the previously saved TRAIN assignment, without
reordering axes or flipping signs. PDF/SVG remain sharp at arbitrary zoom.
"""
from pathlib import Path
import argparse
import csv
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

HERE=Path(__file__).resolve().parent


def build(input_path, comparison_path, output, dpi=180):
    reports=json.loads(Path(input_path).read_text(encoding='utf-8'))
    with Path(comparison_path).open(encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    methods=[r['method'] for r in rows if r['model']=='C1' and r['representation']=='pca_whitened']
    assert len(methods)==len(set(methods))==34
    models=['C1','C2','C3','CR3']
    representations=['pca_whitened','raw_top_standardized']
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(30,68),facecolor='white')
    grid=GridSpec(len(methods),9,figure=fig,width_ratios=[3.6]+[1.7]*8,
                  left=.008,right=.994,bottom=.022,top=.955,hspace=.38,wspace=.27)
    columns=[(representation,model) for representation in representations for model in models]
    max_difference=0.;cells=0
    for index,method in enumerate(methods):
        label=fig.add_subplot(grid[index,0]);label.axis('off')
        prefix,suffix=method.split('__',1) if '__' in method else ('baseline',method)
        label.text(.97,.72,prefix,ha='right',va='center',fontsize=12,
                   color={'probability':'#2166ac','logit':'#9e1b32','ddm':'#5b3c88','baseline':'black'}[prefix],fontweight='bold')
        label.text(.97,.35,'\n'.join(textwrap.wrap(suffix.replace('_',' '),width=34)),
                   ha='right',va='center',fontsize=11)
        for column,(representation,model) in enumerate(columns):
            report=reports[f'{model}/{representation}/{method}']
            stored=np.asarray(report['signed_test_corr'],float)
            matrix=stored.T
            assert matrix.shape==((6,6) if model=='C3' else (4,4))
            assert np.isfinite(matrix).all() and np.max(np.abs(matrix))<=1+1e-12
            max_difference=max(max_difference,float(np.max(np.abs(matrix.T-stored))))
            cells+=matrix.size
            ax=fig.add_subplot(grid[index,column+1])
            ax.imshow(matrix,cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest',aspect='auto')
            names=([r'$\alpha$',r'$\lambda$',r'$\gamma$',r'$\tau$'] if model=='CR3'
                   else [f'$b_{{{j+1}}}$' for j in range(matrix.shape[1])])
            ax.set_xticks(range(matrix.shape[1]),names,fontsize=9)
            ax.set_yticks(range(matrix.shape[0]),[f'u{j+1}' for j in range(matrix.shape[0])],fontsize=8)
            ax.tick_params(length=0,pad=2)
            for parameter,axis in report['train_matching'].items():
                ax.add_patch(Rectangle((int(parameter)-.5,int(axis)-.5),1,1,
                                       fill=False,edgecolor='black',linewidth=1.05))
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    value=matrix[i,j]
                    ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=8,
                            color='white' if abs(value)>.65 else 'black')
            for spine in ax.spines.values():spine.set_linewidth(.35)
            if index==0:
                ax.set_title(model,fontsize=17,fontweight='bold',pad=11)
    for start,title in [(1,'TRAIN PCA-whitened coordinates'),(5,'TRAIN raw-top standardized coordinates')]:
        left=grid[0,start].get_position(fig).x0
        right=grid[0,start+3].get_position(fig).x1
        fig.text((left+right)/2,.975,title,ha='center',fontsize=20,fontweight='bold')
    fig.text(.5,.993,'All historical rotation attempts: signed TEST correlation between u and theta',
             ha='center',va='top',fontsize=24,fontweight='bold')
    fig.text(.5,.986,'34 method labels × 4 models × 2 representations = 272 complete matrices',
             ha='center',va='top',fontsize=16)
    cax=fig.add_axes([.32,.009,.43,.003])
    colorbar=fig.colorbar(ScalarMappable(norm=Normalize(-1,1),cmap='RdBu_r'),cax=cax,orientation='horizontal')
    colorbar.set_label('Pearson r (signed); original axis order; no sign flips',fontsize=13)
    colorbar.ax.tick_params(labelsize=11)
    fig.text(.015,.012,'Black outlines: saved TRAIN matching.\nJoint-Fisher labels are exact choice-only aliases.',fontsize=11,va='center')
    stem=output/'all_methods_four_models_signed_correlations'
    for extension in ['pdf','svg','png']:
        fig.savefig(stem.with_suffix('.'+extension),dpi=dpi,facecolor='white')
    plt.close(fig)
    audit=dict(method_labels=len(methods),models=models,representations=representations,
               matrices=len(methods)*len(columns),cells=cells,color_limits=[-1,1],
               statistic='signed TEST Pearson correlation',row_axis='original u index',
               column_axis='theta parameter',axis_reordering=False,sign_flips=False,
               black_outline='persisted TRAIN Hungarian matching',
               maximum_difference_from_saved_correlations=max_difference)
    stem.with_suffix('.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=HERE/'results/signed_correlations.json')
    parser.add_argument('--comparison',type=Path,default=HERE/'results/comparison.csv')
    parser.add_argument('--output',type=Path,default=HERE.parents[1]/'outputs/choice/historical_transfer/figures')
    parser.add_argument('--dpi',type=int,default=180)
    args=parser.parse_args()
    build(args.input,args.comparison,args.output,args.dpi)
