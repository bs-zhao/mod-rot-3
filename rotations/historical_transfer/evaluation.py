"""Truth access is confined to this post-persistence evaluation module."""
from pathlib import Path
import json
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import corr


def write_json(path,obj):
    def convert(value):
        if isinstance(value,np.ndarray): return value.tolist()
        if isinstance(value,np.generic): return value.item()
        if isinstance(value,Path): return str(value)
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(obj,default=convert,indent=2,allow_nan=False),encoding='utf-8')


def write_csv(path,rows):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def evaluate(source, out, methods):
    # All method matrices and per-method U have been persisted before entering here.
    assert (out/'fit_complete.json').exists()
    for name in methods:
        assert (out/name/'mapping.npz').exists()
    with np.load(source['output_root']/'s1_generate/s1_generate__dataset.npz',allow_pickle=False) as data:
        theta=data['parameters']
    if source['model']=='CR3':
        names=['alpha','lambda_loss','gamma','tau']
    else:
        names=[f'b{i+1}' for i in range(theta.shape[1])]
    tr,te=source['tr'],source['te']
    reports={};rows=[]
    fig,axes=plt.subplots(int(np.ceil(len(methods)/5)),5,figsize=(20,3.5*np.ceil(len(methods)/5)),squeeze=False)
    for index,name in enumerate(methods):
        with np.load(out/name/'mapping.npz') as f:
            u=f['U']
        ct,c=corr(theta[tr],u[tr]),corr(theta[te],u[te])
        i,j=linear_sum_assignment(-ct*ct)
        it,jt=linear_sum_assignment(-c*c)
        off=np.ones(c.shape,bool);off[i,j]=False
        report=dict(parameter_names=names,signed_train_corr=ct,signed_test_corr=c,
                    train_matching=dict(zip(map(str,i),map(int,j))),
                    parameter_r2=dict(zip([names[v] for v in i],(c[i,j]**2).tolist())),
                    mean_r2=float(np.mean(c[i,j]**2)),min_r2=float(np.min(c[i,j]**2)),
                    off_mean_r2=float(np.mean(c[off]**2)),off_max_abs_r=float(np.max(np.abs(c[off]))),
                    diagnostic_test_optimal_mean_r2=float(np.mean(c[it,jt]**2)))
        reports[name]=report
        row=dict(model=source['model'],representation=source['representation']['kind'],method=name,
                 **{key:report[key] for key in ['mean_r2','min_r2','off_mean_r2','off_max_abs_r','diagnostic_test_optimal_mean_r2']})
        rows.append(row)
        write_json(out/name/'evaluation.json',report)
        ax=axes.flat[index];im=ax.imshow(c,vmin=-1,vmax=1,cmap='coolwarm')
        ax.set_title(name.replace('__','\n')+f'\nmean r²={report["mean_r2"]:.3f}',fontsize=8)
        ax.set_yticks(range(len(names)),names,fontsize=7)
        ax.set_xticks(range(u.shape[1]),[f'u{v+1}' for v in range(u.shape[1])],fontsize=7)
        for r in range(c.shape[0]):
            for col in range(c.shape[1]):ax.text(col,r,f'{c[r,col]:.2f}',ha='center',va='center',fontsize=6)
    for ax in axes.flat[len(methods):]:ax.axis('off')
    fig.suptitle(source['model']+' | '+source['representation']['kind']+' | signed TEST r; TRAIN matching',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.98));fig.savefig(out/'all_signed_heatmaps.png',dpi=150);plt.close(fig)
    write_json(out/'evaluation.json',reports);write_csv(out/'comparison.csv',rows)
    return rows
