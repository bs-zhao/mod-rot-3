"""Fixed comparisons from this wave's sealed, separately evaluated results.

This is reporting only. It neither imports latent truth nor changes rotations,
weights, integration budgets or hyperparameters.
"""
import json,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import core as c

def main():
    out=c.HERE/'reports'
    d=pd.read_csv(out/'selected_metrics.csv')
    d['tag']=d.result_path.map(lambda p:p.replace('\\','/').split('/')[3])
    threshold=[]
    for (model,rep),sub in d[d.channel=='prob'].groupby(['model','repeat']):
        base=sub[(sub.experiment=='exp13_reproduce')&(sub.tag=='independent_v1')]
        strict=base[base.name=='A0_strict'];base=(strict if len(strict) else base[base.name=='A0']).iloc[0]
        for _,r in sub.iterrows():
            threshold.append(dict(model=model,repeat=rep,experiment=r.experiment,tag=r.tag,name=r['name'],
                mcc=r.mcc,leakage=r.leakage,max_leakage=r.max_leakage,oracle_angle_max=r.oracle_angle_max,
                frozen_max_leakage=base.max_leakage,absolute_max_leakage_reduction=base.max_leakage-r.max_leakage,
                meets_user_003=bool(base.max_leakage-r.max_leakage>=.03),result_path=r.result_path))
    pd.DataFrame(threshold).to_csv(out/'user_003_criterion.csv',index=False)
    comparisons=[]
    for model in ['CR3','C1','C2','C3']:
        for rep in [1,2]:
            sub=d[(d.model==model)&(d['repeat']==rep)&(d.channel=='prob')]
            specs=[('EXP13','exp13_reproduce','independent_v1','A0_strict' if model=='CR3' else 'A0'),
                   ('TRAIN','exp20A','independent_v1','full_train'),
                   ('P1','exp20B' if model=='CR3' else 'exp20B_confirm','independent_v1',
                    'P1_QMC_4scramble_mean' if model=='CR3' else 'P1_4scramble_mean'),
                   ('P3','exp20B' if model=='CR3' else 'exp20B_confirm','independent_v1',
                    'P3_QMC_4scramble_mean' if model=='CR3' else 'P3_4scramble_mean')]
            # Confirmed pilot sizes differ only where the T-free inner convergence
            # rule required extension. Report their paired sigma=0 separately.
            tags=[t for t in sub.tag.unique() if re.fullmatch(r'confirm\d+',t)]
            tag='verified1024' if model=='CR3' else max(tags,key=lambda t:int(t[7:])) if tags else 'confirm128'
            n=1024 if model=='CR3' else int(tag[7:])
            specs += [('pilot0','exp22',tag,'pilot_sigma0_n1_s')]
            for sigma in ['0.1','0.25']:
                for order in ['s','e']:
                    specs.append((f'pilot{sigma}_{order}','exp22',tag,f'pilot_sigma{sigma}_n{n}_{order}'))
            baseline=None
            for label,exp,tag,name in specs:
                hit=sub[(sub.experiment==exp)&(sub.tag==tag)&(sub.name==name)]
                if len(hit)!=1:continue
                x=hit.iloc[0].to_dict();x['method']=label
                if label=='EXP13':baseline=x
                if baseline:
                    for metric in ['mcc','worst_matched','leakage','max_leakage','oracle_angle_max']:
                        x[metric+'_delta_vs_frozen']=x[metric]-baseline[metric]
                comparisons.append(x)
    comp=pd.DataFrame(comparisons)
    comp.to_csv(out/'four_model_comparison.csv',index=False)
    cols=['model','repeat','method','mcc','worst_matched','leakage','max_leakage','oracle_angle_max','baseline_angle_max']
    (out/'four_model_comparison.md').write_text(comp[cols].to_markdown(index=False,floatfmt='.6f')+'\n',encoding='utf-8')
    # Complete numerical status, including early failed caps; no lucky-prefix rule.
    convergence=[]
    for p in (c.HERE/'exp22/results').glob('*/*/*_cap_status.json'):
        x=json.loads(p.read_text())
        for r in x['rows']:convergence.append(dict(source=str(p.relative_to(c.HERE)),**r))
    pd.DataFrame(convergence).to_csv(out/'smoothing_convergence.csv',index=False)
    for model in ['CR3','C1','C2','C3']:
        sub=comp[comp.model==model];methods=list(dict.fromkeys(sub.method))
        fig,axs=plt.subplots(2,2,figsize=(13,7))
        for ax,metric in zip(axs.flat,['mcc','leakage','max_leakage','oracle_angle_max']):
            for rep,offset in [(1,-.12),(2,.12)]:
                r=sub[sub['repeat']==rep].set_index('method').reindex(methods)
                ax.plot(np.arange(len(methods))+offset,r[metric],'o',label=f'rep{rep}')
            if 'pilot0' in methods:ax.axvline(methods.index('pilot0')-.5,color='gray',ls='--',alpha=.5)
            ax.set(xticks=np.arange(len(methods)),xticklabels=methods,ylabel=metric)
            ax.tick_params(axis='x',rotation=45);ax.grid(alpha=.2)
        axs.flat[0].legend();fig.suptitle(f'{model}: fixed comparisons; pilot methods have their own matched baseline')
        fig.tight_layout();fig.savefig(out/f'comparison_{model}.png',dpi=150);plt.close(fig)
    print(comp[cols].to_string(index=False))

if __name__=='__main__':main()
