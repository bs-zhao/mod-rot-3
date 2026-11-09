"""Read-only comparison, figures and inventory for the completed experiments."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from summarize import main as inventory

HERE=Path(__file__).resolve().parent

def find(label,model,source,estimator,metric,channel='logit',degree=None):
    runs=[]
    for path in (HERE/'results'/label).rglob('summary.json'):
        value=json.loads(path.read_text());a=value['args']
        if (a['model'],a['source'],a['estimator'],a['metric'],a['channel'])!=(model,source,estimator,metric,channel): continue
        if degree is not None and a.get('degree')!=degree: continue
        runs.append((value,path))
    if len(runs)!=1: raise ValueError(f'Expected one result: {label} {model} {source} {estimator}: {len(runs)}')
    return runs[0]

def main():
    inventory()
    specs=[('C1','ux_column_entropy','round06_loworder','polynomial','round06_loworder'),
           ('C2','signature_l1','round03','polynomial','round04'),
           ('C3','signature_l1','round03','polynomial','round04'),
           ('CR3','conditional_ratios','round06','stimulus','round06')]
    rows=[];fig,axes=plt.subplots(2,2,figsize=(11,8.3))
    heat,ha=plt.subplots(1,4,figsize=(16,4.1))
    for ax,hax,(model,metric,glabel,est,dlabel) in zip(axes.flat,ha,specs):
        direct,dp=find(glabel,model,'formula','direct',metric)
        numeric,npth=find(glabel,model,'formula',est,metric)
        decoder,decpath=find(dlabel,model,'decoder',est,metric)
        def curves(path):
            with (path.parent/'planes.csv').open() as f:return list(csv.DictReader(f))
        a,b=curves(dp),curves(npth)
        exact=np.array([float(x['loss']) for x in a]);approx=np.array([float(x['loss']) for x in b])
        cc=float(np.corrcoef(exact,approx)[0,1]);relative=float(np.linalg.norm(exact-approx)/max(np.linalg.norm(exact),1e-30))
        for i,j in sorted(set((x['i'],x['j']) for x in a)):
            ids=[k for k,x in enumerate(a) if x['i']==i and x['j']==j]
            angles=[float(a[k]['angle']) for k in ids]
            ax.plot(angles,exact[ids],lw=1,alpha=.8)
            ax.plot(angles,approx[ids],lw=.7,ls='--',alpha=.6,color='black')
        ax.set(title=f'{model}: {metric}\nG1 exact vs G2 WY, r={cc:.5f}',xlabel='Plane angle (degrees)',ylabel='Loss')
        ev=decoder['evaluation'];cm=np.array(ev['signed']);perm=np.argsort(ev['matching']);cm=cm[perm];cm*=np.sign(np.diag(cm))[:,None]
        image=hax.imshow(cm,vmin=-1,vmax=1,cmap='coolwarm')
        hax.set_title(f'{model}: original TEST\nmatched={ev["matched_mean"]:.3f}',fontsize=11)
        hax.set_xticks(range(len(cm)),[f'T{k+1}' for k in range(len(cm))]);hax.set_yticks(range(len(cm)),[f'U{k+1}' for k in range(len(cm))])
        for i in range(len(cm)):
            for j in range(len(cm)):hax.text(j,i,f'{cm[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(cm[i,j])>.6 else 'black')
        oracle_path=next((HERE/'results/oracles').glob(f'{model}_*pca0.json'));oracle=json.loads(oracle_path.read_text())
        row=dict(model=model,metric=metric,g1_strict=direct['gap1_strict_pass'],g2_axis_min=numeric['best']['axis_min'],
                 g2_curve_correlation=cc,g2_curve_relative_error=relative,
                 test_matched=ev['matched_mean'],test_worst=ev['matched_min'],test_offmax=ev['off_max'],
                 supervised_rotation_matched=oracle['rotation_test']['matched_mean'],discovery_result=str(decpath.relative_to(HERE)))
        rows.append(row)
    fig.tight_layout();fig.savefig(HERE/'results/gap12_landscapes.png',dpi=160);plt.close(fig)
    heat.tight_layout();heat.savefig(HERE/'results/decoder_test_correlations.png',dpi=160);plt.close(heat)
    with (HERE/'results/selected_comparison.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
