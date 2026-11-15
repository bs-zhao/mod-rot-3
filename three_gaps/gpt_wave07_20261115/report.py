"""Rebuild only from completed own results; no access to raw truth."""
from pathlib import Path
import re,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import core as c

def main():
    tables=[]
    for p in c.HERE.glob('*/results/*/*/*/metrics.csv'):
        if not (p.parent/'COMPLETE.json').exists():continue
        d=pd.read_csv(p);d['result_path']=str(p.relative_to(c.HERE));tables.append(d)
    if not tables:return
    allrows=pd.concat(tables,ignore_index=True);out=c.HERE/'reports';out.mkdir(exist_ok=True)
    allrows.to_csv(out/'all_start_metrics.csv',index=False)
    d=allrows[allrows.selected].copy();d.to_csv(out/'selected_metrics.csv',index=False)
    moment_rows=[]
    for _,row in d[d.experiment=='exp20A'].iterrows():
        if not re.match(r'N\d+_seed\d+_fixed$',row['name']):continue
        cfg=json.loads((c.HERE/row.result_path).with_name('config.json').read_text())['extra']
        moment_rows.append(dict(model=row.model,repeat=row['repeat'],channel=row.channel,name=row['name'],
            **{k:cfg[k] for k in ['N','SX_relative','T4_relative','landscape_spearman','landscape_pearson']},
            full_train_angle_max=row.full_train_angle_max))
    moments=pd.DataFrame(moment_rows);moments.to_csv(out/'subject_moment_convergence.csv',index=False)
    for (model,channel),sub in moments.groupby(['model','channel']):
        fig,axs=plt.subplots(2,2,figsize=(10,7))
        for rep in sorted(sub['repeat'].unique()):
            stats=sub[sub['repeat']==rep].groupby('N')[['SX_relative','T4_relative','landscape_spearman','full_train_angle_max']].agg(['mean','std'])
            for ax,m in zip(axs.flat,['SX_relative','T4_relative','landscape_spearman','full_train_angle_max']):
                ax.errorbar(stats.index,stats[(m,'mean')],yerr=stats[(m,'std')].fillna(0),fmt='o-',label=f'rep{rep}')
                ax.set(xlabel='TRAIN subjects',ylabel=m,xscale='log');ax.grid(alpha=.2)
        axs.flat[0].legend();fig.suptitle(f'{model} {channel}: full TRAIN reference; 5 seeds, fixed covariance')
        fig.tight_layout();fig.savefig(out/f'moment_convergence_{model}_{channel}.png',dpi=150);plt.close(fig)
    # Consistent grids show matching, nonmatching, and rotation angle together.
    for model in d.model.unique():
        sub=d[(d.model==model)&(d.experiment=='exp20A')].copy()
        if sub.empty:continue
        fig,axs=plt.subplots(2,3,figsize=(14,7))
        measures=['mcc','worst_matched','leakage','max_leakage','oracle_angle_max','baseline_angle_max']
        for rep in sorted(sub['repeat'].unique()):
            for mode,style in [('fixed','-'),('native','--')]:
                x=sub[(sub['repeat']==rep)&sub.name.str.match('N.*_'+mode)].copy()
                x['N']=x.name.str.extract(r'N(\d+)').astype(int)
                stats=x.groupby('N')[measures].agg(['mean','std'])
                for ax,m in zip(axs.flat,measures):
                    ax.errorbar(stats.index,stats[(m,'mean')],yerr=stats[(m,'std')].fillna(0),fmt='o'+style,ms=3,label=f'rep{rep} {mode}')
            full=sub[(sub['repeat']==rep)&(sub.name=='full_train')]
            if len(full):
                for ax,m in zip(axs.flat,measures):ax.scatter([3500],[full.iloc[0][m]],marker='*',s=65)
        for ax,m in zip(axs.flat,measures):ax.set(xlabel='TRAIN subjects (star = full TRAIN)',ylabel=m,xscale='log');ax.grid(alpha=.2)
        axs.flat[0].legend(fontsize=8);fig.suptitle(f'{model}: independent EXP20A, 5 nested seeds (mean +/- SD)')
        fig.tight_layout();fig.savefig(out/f'exp20A_{model}.png',dpi=160);plt.close(fig)
    # Audit complete vs unevaluated explicitly; completion is not a success label.
    coverage=[]
    for exp in ['exp13_reproduce','exp20A','exp20B','exp20B_confirm','exp22','exp21','exp23']:
        n=len(list((c.HERE/exp/'results').glob('*/*/*/COMPLETE.json')))
        m=len(d[d.experiment==exp]);coverage.append(dict(experiment=exp,blind_completed=n,evaluated=m))
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    cols=['experiment','model','repeat','channel','name','mcc','worst_matched','leakage','max_leakage','oracle_angle_max','baseline_angle_max']
    primary=d[~d.name.str.match(r'N\d+_seed')].copy()
    (out/'tables.md').write_text('# Independently evaluated completed candidates\n\n'+primary[cols].to_markdown(index=False,floatfmt='.5f')+'\n',encoding='utf-8')
    print(pd.DataFrame(coverage).to_string(index=False))
    diagnostics(out,d)

def diagnostics(out,d):
    """Post-hoc outputs, derived only from completed rotations and evaluations."""
    axes=[];runtime=[];manifest=[];prediction=[]
    for _,row in d.iterrows():
        p=c.HERE/row.result_path;folder=p.parent
        cp=folder/'correlations.npz';rp=folder/'rotations.npz'
        if cp.exists():
            with np.load(cp) as z:
                M=abs(z['signed_matched']);oa=z['oracle_angles'];ba=z['baseline_angles']
                for i in range(len(M)):
                    off=np.delete(M[i],i)
                    axes.append(dict(result_path=row.result_path,axis=i,matched=M[i,i],off_mean=off.mean(),off_max=off.max(),
                        oracle_angle=oa[i] if len(oa) else None,baseline_angle=ba[i] if len(ba) else None))
        # Retain D_R for every start, without changing immutable rotations.npz.
        with np.load(rp) as z:
            R=z['R'].astype(float);C=z['C_ref'];D=np.sqrt(np.einsum('kia,ab,kib->ki',R,C,R))
            c.save_npz(folder/'axis_scales.npz',D_R=D,A=D[:,:,None]*np.linalg.inv(R).transpose(0,2,1))
        dp=folder/'diagnostics.json'
        if dp.exists():
            x=json.loads(dp.read_text());runtime.append(dict(result_path=row.result_path,
                seconds=x.get('seconds'),peak_cuda_allocated_bytes=x.get('peak_cuda_allocated_bytes'),scope='candidate diagnostics'))
    pd.DataFrame(axes).to_csv(out/'per_axis_metrics.csv',index=False)
    # Job wall times include decoder queries, optimization and reporting; nested
    # candidate times overlap, so never sum these two scopes as a total.
    for p in (c.OUT/'logs').rglob('*.status.json'):
        x=json.loads(p.read_text());command=p.with_name(p.name.replace('.status.json','.command.json'))
        cmd=json.loads(command.read_text()) if command.exists() else {}
        argv=x.get('argv',cmd.get('argv',[]))
        runtime.append(dict(job=str(p.relative_to(c.OUT)),seconds=x.get('seconds'),
            returncode=x.get('returncode'),scope='subprocess wall time',argv=json.dumps(argv)))
        manifest.append(dict(job=str(p.relative_to(c.OUT)),command=cmd,status=x))
    pd.DataFrame(runtime).to_csv(out/'runtime.csv',index=False)
    c.dump(out/'execution_manifest.json',manifest)
    for p in (c.HERE/'exp22/results').glob('*/*/*_validation_prediction.json'):
        for row in json.loads(p.read_text()):
            prediction.append(dict(source=str(p.relative_to(c.HERE)),**{k:v for k,v in row.items() if k!='validation_subjects'}))
    pd.DataFrame(prediction).to_csv(out/'validation_prediction.csv',index=False)
    # Fixed representative names, not chosen by TEST performance.
    keep=(d.name.isin(['A0_strict','full_train','P1_QMC_4scramble_mean','P3_QMC_4scramble_mean','P1_4scramble_mean','P3_4scramble_mean'])|
          d.name.str.contains(r'full_sigma0\.1_n64|pilot_n1024_|ambient_gap4|hybrid_h0\.25|joint_.*sigma0\.25_n512',regex=True))
    for _,row in d[keep].iterrows():
        folder=(c.HERE/row.result_path).parent;fp=folder/'oracle_plane_curves.csv'
        if not fp.exists():continue
        x=pd.read_csv(fp);groups=list(x.groupby(['i','j']));nc=3;nr=int(np.ceil(len(groups)/nc))
        fig,axs=plt.subplots(nr,nc,figsize=(11,2.4*nr),squeeze=False)
        for ax,((i,j),g) in zip(axs.flat,groups):
            ax.plot(g.degrees_from_oracle,g.loss);ax.axvline(0,color='black',alpha=.3)
            ax.set(xlabel='degrees from Oracle (diagnostic only)',ylabel='loss',title=f'plane {i},{j}');ax.grid(alpha=.2)
        for ax in list(axs.flat)[len(groups):]:ax.set_visible(False)
        fig.suptitle(f'{row.model} rep{row["repeat"]} {row["name"]}',fontsize=9)
        fig.tight_layout();fig.savefig(folder/'oracle_planes.png',dpi=130);plt.close(fig)

if __name__=='__main__':main()
