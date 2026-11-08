"""Read four completed Stage-2 refiner reports; NO training or refitting."""
from pathlib import Path
import argparse
import csv
import json

MODELS={'C1':4,'C2':4,'C3':6,'CR3':4}
DEFAULT_BETA={'C1':0.0005,'C2':0.0005,'C3':0.0005,'CR3':0.00005}
ROOT=Path(__file__).resolve().parents[2]
def number_tag(v):
    return f'{float(v):g}'.replace('-','m').replace('.','p').replace('+','')
def condition(beta,sparse,lo,hi,d):
    return (f'beta{number_tag(beta)}_lambda{number_tag(sparse)}_'
            f'ctx{number_tag(lo)}to{number_tag(hi)}_L{d}')

def aggregate(root,*,beta=.0005,cr3_beta=.00005,sparse=0.,lo=.25,hi=.80,repeat=1):
    root=Path(root)
    rows=[];parameter_rows=[]
    for model,d in MODELS.items():
        model_beta=cr3_beta if model=='CR3' else beta
        folder=(root/model/condition(model_beta,sparse,lo,hi,d)/
                f'repeat{repeat:02d}'/'stage2_joint_orthogonal_scaled')
        path=folder/'summary.json'
        if not path.exists():
            raise FileNotFoundError(f'First run Stage 2 for {model}: {path}')
        report=json.loads(path.read_text(encoding='utf-8'))
        if report['model']!=model:
            raise ValueError(f'Stage 2 model mismatch for {path}')
        base=report['models']['direct_ols_oracle']['test_metrics']
        for method,info in report['models'].items():
            m=info['test_metrics']
            rows.append(dict(model=model,dim=d,beta=model_beta,method=method,
                mean_abs_diag_r=m['abs_diag_mean'],
                max_abs_offdiag_r=m['abs_offdiag_max'],
                mean_test_r2=m['mean_r2'],
                delta_r2_from_oracle=m['mean_r2']-base['mean_r2'],
                delta_max_offdiag_from_oracle=m['abs_offdiag_max']-base['abs_offdiag_max']))
        file=folder/'per_parameter.csv'
        with file.open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                parameter_rows.append(row)
    root.mkdir(parents=True,exist_ok=True)
    with (root/'four_models_stage2_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    with (root/'four_models_stage2_per_parameter.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(parameter_rows[0]))
        writer.writeheader();writer.writerows(parameter_rows)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT/'outputs'/'math'/'09_procrustes_vs_ols')
    p.add_argument('--beta',type=float,default=.0005)
    p.add_argument('--cr3-beta',type=float,default=.00005)
    p.add_argument('--lambda-sparse',type=float,default=0.)
    p.add_argument('--context-min',type=float,default=.25)
    p.add_argument('--context-max',type=float,default=.80)
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--no-plots',action='store_true')
    a=p.parse_args()
    rows=aggregate(a.root,beta=a.beta,cr3_beta=a.cr3_beta,
                   sparse=a.lambda_sparse,lo=a.context_min,
                   hi=a.context_max,repeat=a.repeat)
    print('Saved four-model comparison:',a.root/'four_models_stage2_comparison.csv')
    for model in MODELS:
        print(model)
        for row in rows:
            if row['model']==model:
                print(f"  {row['method']:32s} R2={row['mean_test_r2']:.4f} "
                      f"diag|r|={row['mean_abs_diag_r']:.4f} "
                      f"max wrong r={row['max_abs_offdiag_r']:.4f}")
    if a.no_plots:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=list(dict.fromkeys(r['method'] for r in rows))
    x=range(len(MODELS))
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,key,title in zip(axes,
                            ('mean_test_r2','mean_abs_diag_r','max_abs_offdiag_r'),
                            ('TEST mean R2','Mean diagonal |r|','Maximum wrong-parameter |r|')):
        for method in methods:
            values=[next(r[key] for r in rows if r['model']==model and r['method']==method)
                    for model in MODELS]
            ax.plot(x,values,'o-',label=method)
        ax.set_xticks(list(x),list(MODELS))
        ax.set_title(title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    path=a.root/'four_models_stage2_comparison.png'
    fig.savefig(path,dpi=165);plt.close(fig)
    print('Saved:',path)

if __name__=='__main__':
    main()
