"""Additional postprocessing only: historical coordinates and cross-Y landscapes."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D


def historical(path):
    data=pd.read_csv(path/'metric_values.csv')
    metrics=json.loads((path/'catalog.json').read_text())['metric_names']
    selected=data[data.family.isin(['reference','oracle_control','historical'])]
    order=selected[['case_id','family']].drop_duplicates().sort_values('family',key=lambda s:s.map({'reference':0,'oracle_control':1,'historical':2}),kind='stable').case_id.tolist()
    with PdfPages(path/'historical_metric_atlas.pdf') as pdf:
        for response,g in selected.groupby('response',sort=False):
            g=g.set_index('case_id').loc[order]
            raw=g[metrics].to_numpy().T
            delta=raw-raw[:,0,None]
            valid=np.isfinite(raw).any(1)
            stats=np.where(valid[:,None],raw,0.)
            scale=np.nanquantile(stats,.75,axis=1)-np.nanquantile(stats,.25,axis=1)
            scale=np.maximum(scale,1e-8*np.maximum(1,np.nanmax(np.abs(stats),axis=1)))
            visual=np.clip(delta/scale[:,None],-3,3)
            fig=plt.figure(figsize=(28,22))
            grid=fig.add_gridspec(2,1,height_ratios=[1,14],hspace=.06,left=.22,right=.97,bottom=.05,top=.94)
            ax=fig.add_subplot(grid[0]);colors=['#228833' if v=='reference' else '#ee9933' if v=='oracle_control' else '#777777' for v in g.family]
            ax.bar(range(len(g)),g.recovery_mean_r2,color=colors,width=.85);ax.set_ylim(0,1)
            ax.set_ylabel('TEST theta r2',fontsize=9);ax.set_xticks([]);ax.tick_params(labelsize=8)
            ax=fig.add_subplot(grid[1]);im=ax.imshow(np.ma.masked_invalid(visual),aspect='auto',cmap='coolwarm',vmin=-3,vmax=3)
            ax.set_yticks(range(len(metrics)),metrics,fontsize=7)
            labels=['U*' if family=='reference' else 'theta' if family=='oracle_control' else f'H{int(i):04d}' for i,family in zip(g.index,g.family)]
            ax.set_xticks(range(len(g)),labels,rotation=90,fontsize=5)
            fig.colorbar(im,ax=ax,shrink=.35,pad=.008,label='(metric - U* metric) / historical IQR; clipped +/-3')
            fig.suptitle(f'{path.name} | {response} | all saved historical coordinates; case names in case_metadata.csv',fontsize=13)
            pdf.savefig(fig);plt.close(fig)
    scan=data[data.family=='plane'];planes=list(scan.plane.unique())
    fig,ax=plt.subplots(figsize=(10,3));ax.axis('off')
    handles=[Line2D([0],[0],c=plt.cm.tab20(i%20),lw=2,label=plane) for i,plane in enumerate(planes)]
    handles += [Line2D([0],[0],c='black',ls='--',label='joint path'),Line2D([0],[0],c='red',marker='o',ls='',label='U*')]
    ax.legend(handles=handles,loc='center',ncol=5,frameon=False,fontsize=9)
    ax.set_title(path.name+' | color key for every curves_<Y>.pdf')
    fig.tight_layout();fig.savefig(path/'plane_color_key.png',dpi=180);plt.close(fig)


def comparison_heatmap(run,filename,value,title,output):
    frame=pd.read_csv(run/filename)
    metrics=json.loads((run/'C1/catalog.json').read_text())['metric_names']
    models=['C1','C2','C3','CR3'];responses=['observed_choice','decoder_choice','decoder_logit','true_logit','true_logit_noise']
    matrix=np.array([[frame[(frame.model==model)&(frame.response==response)&(frame.metric==metric)][value].iloc[0] for model in models for response in responses] for metric in metrics])
    fig,ax=plt.subplots(figsize=(15,21));im=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_yticks(range(len(metrics)),metrics,fontsize=7)
    ax.set_xticks(range(20),[model+'\n'+response.replace('_','\n') for model in models for response in responses],fontsize=7)
    for i in [4.5,9.5,14.5]:ax.axvline(i,c='black',lw=.8)
    ax.set_title(title,fontsize=11);fig.colorbar(im,ax=ax,shrink=.35,label='Spearman rho; blank = undefined')
    fig.tight_layout();fig.savefig(run/output,dpi=180);plt.close(fig)


def curve_grid(run,metric):
    models=['C1','C2','C3','CR3'];responses=['observed_choice','decoder_choice','decoder_logit','true_logit','true_logit_noise']
    all_planes=[f'u{i}_u{j}' for i in range(1,7) for j in range(i+1,7)]
    color={name:plt.cm.tab20(index) for index,name in enumerate(all_planes)}
    fig,axes=plt.subplots(4,5,figsize=(22,14))
    for row,model in enumerate(models):
        frame=pd.read_csv(run/model/'metric_values.csv')
        for col,response in enumerate(responses):
            ax=axes[row,col];g=frame[frame.response==response]
            for plane,part in g[g.family=='plane'].groupby('plane',sort=False):
                part=part.sort_values('angle');ax.plot(part.angle,part[metric],c=color[plane],lw=.65,alpha=.6)
            joint=g[g.family=='joint_path'].sort_values('angle');ax.plot(joint.angle,joint[metric],'k--',lw=1.5)
            base=g[g.family=='reference'][metric].iloc[0];ax.scatter([0],[base],c='red',s=18,zorder=10)
            ax.axvline(0,c='gray',lw=.5);ax.set_xticks([-90,-45,0,45,90]);ax.tick_params(labelsize=7)
            if row==0:ax.set_title(response,fontsize=10)
            if col==0:ax.set_ylabel(model+'\nmetric value',fontsize=11)
            if row==3:ax.set_xlabel('Rotation angle (degrees)',fontsize=9)
    handles=[Line2D([0],[0],c=color[name],label=name) for name in all_planes]
    handles += [Line2D([0],[0],c='black',ls='--',label='joint path'),Line2D([0],[0],c='red',marker='o',ls='',label='U*')]
    fig.legend(handles=handles,loc='lower center',ncol=9,fontsize=8,frameon=False)
    fig.suptitle(metric+' | all four models and five fixed Y versions\nDescriptive landscapes; proxy structure is not an oracle derivative',fontsize=13)
    fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(run/(metric+'__four_models_five_Y.png'),dpi=180)
    fig.savefig(run/(metric+'__four_models_five_Y.pdf'));plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True,type=Path);args=p.parse_args()
    run=args.run
    for model in ['C1','C2','C3','CR3']:historical(run/model)
    comparison_heatmap(run,'response_landscape_agreement.csv','spearman_landscape_vs_true_logit',
        'Does each Y preserve the true-logit metric landscape? | same reference rotations',
        'response_landscape_agreement.png')
    comparison_heatmap(run,'metric_recovery_associations_DIAGNOSTIC.csv','spearman_with_parameter_recovery',
        'Posthoc metric vs theta recovery | diagnostic, not selected objectives',
        'metric_vs_recovery_DIAGNOSTIC.png')
    for metric in ['proxy_module_code','codec_given_otherU_X__axis_rms']:curve_grid(run,metric)
    (run/'plot_comparisons_provenance.json').write_text(json.dumps(dict(
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Postprocessing only; no metric definition or numerical outcome changed'),indent=2),encoding='utf-8')
    print('Additional comparison figures complete',flush=True)


if __name__=='__main__':main()
