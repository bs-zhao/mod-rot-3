"""Read four already-computed runs; never fits models or reruns an experiment."""
import argparse
import csv
import json
from pathlib import Path

MODELS = {'C1':4, 'C2':4, 'C3':6, 'CR3':4}
ROOT = Path(__file__).resolve().parents[2]


def tag(beta, sparse, cmin, cmax, dim):
    fmt=lambda v:f'{v:g}'.replace('-','m').replace('.','p').replace('+','')
    return f'beta{fmt(beta)}_lambda{fmt(sparse)}_ctx{fmt(cmin)}to{fmt(cmax)}_L{dim}'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--beta',type=float,default=.0005)
    p.add_argument('--lambda-sparse',type=float,default=0.)
    p.add_argument('--context-min',type=float,default=.25)
    p.add_argument('--context-max',type=float,default=.80)
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--root',type=Path,default=ROOT/'outputs'/'math'/'09_procrustes_vs_ols')
    args=p.parse_args()
    rows=[]
    for model, dim in MODELS.items():
        path=args.root/model/tag(args.beta,args.lambda_sparse,args.context_min,args.context_max,dim)/f'repeat{args.repeat:02d}'/'summary.json'
        if not path.is_file():
            raise FileNotFoundError(f'Run {model} first: {path}')
        report=json.loads(path.read_text(encoding='utf-8'))
        for method, scores in report['test_primary'].items():
            rows.append(dict(model=model,dim=dim,method=method,**scores))
    args.root.mkdir(parents=True,exist_ok=True)
    path=args.root/'four_models_test_comparison.csv'
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Wrote',path)
    for model in MODELS:
        items=[r for r in rows if r['model']==model]
        print(model)
        for r in items:
            print(f"  {r['method']:34s} diag|r|={r['abs_diag_mean']:.4f} "
                  f"offdiag-max={r['abs_offdiag_max']:.4f} R2={r['mean_r2']:.4f}")
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib absent; skipped optional figure')
        return
    methods=list(dict.fromkeys(r['method'] for r in rows))
    xs=range(len(methods))
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for ax,key,caption in zip(axes,('abs_diag_mean','abs_offdiag_max','mean_r2'),
                              ('Mean |diag r|','Max |offdiag r|','Mean held-out R2')):
        width=.18
        for i,model in enumerate(MODELS):
            found={r['method']:r for r in rows if r['model']==model}
            ax.bar([j+(i-1.5)*width for j in xs],[found[m][key] for m in methods],
                   width=width,label=model)
        ax.set_xticks(list(xs),[m.replace('procrustes','proc') for m in methods],rotation=45,ha='right',fontsize=7)
        ax.set_title(caption);ax.grid(axis='y',alpha=.2)
    axes[0].legend()
    fig.tight_layout()
    out=args.root/'four_models_test_comparison.png'
    fig.savefig(out,dpi=160);plt.close(fig)
    print('Wrote',out)

if __name__=='__main__':
    main()
