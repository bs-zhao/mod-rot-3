"""Read-only result inventory. Does not select a decoder loss using TEST labels."""
import csv
import json
from pathlib import Path
import numpy as np

def main():
    root=Path(__file__).resolve().parent/'results'; rows=[]
    for path in sorted(root.rglob('summary.json')):
        s=json.loads(path.read_text(encoding='utf8'));a=s['args'];b=s['best']
        rows.append(dict(round=a['label'],model=a['model'],pca=a['pca'],source=a['source'],channel=a['channel'],
                         estimator=a['estimator'],metric=a['metric'],seed=a['seed'],samples=a['samples'],
                         degree=a.get('degree'),ridge=a.get('ridge'),structure_link=a.get('structure_link','native'),
                         train_rows=a.get('train_rows'),train_subjects=a.get('train_subjects'),
                         strict=s['gap1_strict_pass'],near10=bool(b['axis_min']>np.cos(np.deg2rad(10)) and min(s['plane_ranges'])>1e-5*(1+abs(s['truth_loss']))),
                         axis_min=b['axis_min'],truth=s['truth_loss'],best=b['loss'],improvement=s['truth_loss']-b['loss'],
                         failed_optimizers=sum(not x['success'] for x in s['solutions']),invariance=s['invariance_max'],
                         test_matched=s['evaluation']['matched_mean'] if s['evaluation'] else '',
                         test_worst=s['evaluation']['matched_min'] if s['evaluation'] else '',
                         test_offmax=s['evaluation']['off_max'] if s['evaluation'] else ''))
    if rows:
        with (root/'inventory.csv').open('w',newline='',encoding='utf8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for model in ['C1','C2','C3','CR3']:
        subset=[x for x in rows if x['model']==model and x['source']=='formula']
        print(model,'runs',len(subset),'strict',sum(x['strict'] for x in subset),'near10',sum(x['near10'] for x in subset))

if __name__=='__main__': main()
