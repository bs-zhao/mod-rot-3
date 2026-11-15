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
    for exp in ['exp13_reproduce','exp20A','exp20B','exp22','exp21','exp23']:
        n=len(list((c.HERE/exp/'results').glob('*/*/*/COMPLETE.json')))
        m=len(d[d.experiment==exp]);coverage.append(dict(experiment=exp,blind_completed=n,evaluated=m))
    pd.DataFrame(coverage).to_csv(out/'coverage.csv',index=False)
    cols=['experiment','model','repeat','channel','name','mcc','worst_matched','leakage','max_leakage','oracle_angle_max','baseline_angle_max']
    primary=d[~d.name.str.match(r'N\d+_seed')].copy()
    (out/'tables.md').write_text('# Independently evaluated completed candidates\n\n'+primary[cols].to_markdown(index=False,floatfmt='.5f')+'\n',encoding='utf-8')
    print(pd.DataFrame(coverage).to_string(index=False))

if __name__=='__main__':main()
