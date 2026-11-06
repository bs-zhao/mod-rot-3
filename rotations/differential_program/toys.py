"""Analytic gates, with angle curves before benchmark theta evaluation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'np'/'src')]
import json
import numpy as np
import torch
from .objectives import ProgramObjective,RecurrenceObjective,rotation
from .jets import directional
from itertools import combinations_with_replacement,permutations


BLOCKS=[(1,0),(2,0),(3,0),(1,1),(1,2),(2,1)]


def analytic_blocks(fun,q,x):
    k=q.shape[1];d=x.shape[1]
    points=np.concatenate((np.repeat(q,len(x),axis=0),np.tile(x,(len(q),1))),axis=1)
    value=torch.tensor(points,dtype=torch.float64)
    blocks={}
    for r,s in BLOCKS:
        tensor=np.zeros((len(value),)+(k,)*r+(d,)*s)
        for qi in combinations_with_replacement(range(k),r):
            for xi in combinations_with_replacement(range(d),s):
                derivative=directional(torch.vmap(fun),value,qi+tuple(k+j for j in xi)).detach().numpy()
                for qp in set(permutations(qi)):
                    for xp in set(permutations(xi)):tensor[(slice(None),)+qp+xp]=derivative
        if not np.isfinite(tensor).all():raise ValueError('Invalid analytic derivative tensor')
        blocks[r,s]=tensor
    return blocks


def run_toys():
    torch.set_num_threads(1)
    rng=np.random.default_rng(91)
    cases={
        'power':(lambda v:v[0]*torch.exp(v[1]*v[2]),np.column_stack((rng.uniform(.5,1.5,96),rng.uniform(-1,1,96))),np.linspace(-1,1,48)[:,None]),
        'bilinear':(lambda v:v[0]*v[1]*v[2],rng.normal(size=(96,2)),rng.uniform(-1,1,(48,1))),
        'linear_stimulus':(lambda v:v[0]+v[1]*v[2],rng.normal(size=(96,2)),rng.uniform(-1,1,(48,1))),
    }
    results={};curves=[]
    for name,(fun,q,x) in cases.items():
        b=analytic_blocks(fun,q,x);p=ProgramObjective(b,2,1);rec=RecurrenceObjective(b,2,len(q),len(x))
        series={key:[] for key in ['group_l1','module_code','recurrence']}
        for deg in np.linspace(-90,90,181):
            a=rotation(torch.tensor([np.deg2rad(deg)]),2)
            vals={key:float(p.module_loss(a,key)) for key in ['group_l1','module_code']}
            vals['recurrence']=float(rec.loss(a))
            for key,value in vals.items():series[key].append(value)
            curves.append(dict(toy=name,angle=float(deg),**vals))
        results[name]={key:dict(identity=values[90],min_loss=min(values),best_angle=float(np.linspace(-90,90,181)[int(np.argmin(values))]),
                                at_45=values[135],range=max(values)-min(values)) for key,values in series.items()}
    return results,curves


if __name__=='__main__':
    results,curves=run_toys();out=ROOT/'outputs'/'research'/'differential_program'/'toys_v1';out.mkdir(parents=True,exist_ok=False)
    (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    import csv
    with (out/'curves.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(curves[0]));w.writeheader();w.writerows(curves)
    print(json.dumps(results,indent=2))
