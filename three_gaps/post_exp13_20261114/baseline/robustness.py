"""PCA/unit sensitivity and available CR3 beta checkpoints, in isolated outputs."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as C
from common.decoder import decoder_h
from common.derivs import base_derivatives

def variant(c,pca=False,units=False):
    c=dict(c); c['pca']=int(pca)
    if pca:
        # Full TRAIN PCA, fixed before sampling/search; no dimension removal here.
        raw=torch.load(c['cache_path'],weights_only=False)
        ev,vec=np.linalg.eigh(np.cov(c['V'][raw['tr']].T)); P=C.tensor(vec[:,np.argsort(ev)[::-1]].T.copy())
        c['V']=c['V']@P.numpy().T; c['Vtrain']=C.tensor(c['V'][c['ids']])
        c['g']=torch.einsum('ia,ska->ski',P,C.tensor(c['g']))
        c['Htx']=torch.einsum('ia,skaj->skij',P,C.tensor(c['Htx']))
        c['Htt']=torch.einsum('ia,skab,jb->skij',P,C.tensor(c['Htt']),P)
        c['diagnostic_oracle']=C.oracle({k:v for k,v in c.items() if k!='diagnostic_oracle'})@P.numpy().T
        base=C.load(c['model'],c['repeat'],c['channel'])
        c['baseline_rotation_override']=C.baseline_rotation(base)@P.numpy().T
        c['PCA_matrix']=P.numpy()
    if units:
        factors=torch.ones(c['X'].shape[1],dtype=torch.float64); factors[0]=100.; factors[1]=.01
        c['X']=c['X']*factors; c['Htx']=c['Htx']/factors
        c['X_unit_factors']=factors
    return c

def beta_cache(beta,channel):
    original=C.load('CR3',1,channel)
    data,truth=C.io.load_all('CR3',beta,4,0,1)
    out=C.ROOT/'outputs/three_gaps/post_exp13_20261114/beta_fields'; out.mkdir(parents=True,exist_ok=True)
    path=out/f"CR3_{data['condition']}_{channel}.pt"
    ids=original['ids']
    if not np.isin(ids,data['train_ids']).all():
        ids=np.sort(np.random.default_rng(20261108).choice(data['train_ids'],300,replace=False))
    if not path.exists():
        dev=C.io.pick_device('auto'); net,_=C.io.load_decoder(data['paths']['checkpoint'],dev)
        h=decoder_h(net,data['mu_z'],data['Minv'],channel)
        b=base_derivatives(h,C.tensor(data['V'][ids]).to(dev),original['X'].to(dev),which=('g','Htt','Htx'),chunk=1024)
        raw={k:v.cpu() for k,v in b.items()}
        raw.update(ids=ids,X=original['X'],V=data['V'],Tst=truth.std(),te=data['test_ids'],tr=data['train_ids'])
        torch.save(raw,path)
    raw=torch.load(path,weights_only=False)
    c={k:raw[k] for k in ('g','Htt','Htx','h','ids','X','V')}
    oracle_path=C.ROOT/'outputs/three_gaps/exp00'/f"CR3_{data['condition']}_pca0"/'R_oracle_raw.npy'
    c.update(model='CR3',repeat=1,channel=channel,Vtrain=C.tensor(raw['V'][raw['ids']]),cache_path=str(path),beta=beta,
             diagnostic_oracle=np.load(oracle_path),checkpoint_path=data['paths']['checkpoint'],
             baseline_rotation_override=np.eye(4),baseline_label='New NP identity',
             checkpoint_sha256=C.io.sha256(data['paths']['checkpoint']))
    return c

def main(a):
    for model in C.COND:
        for rep in (1,2):
            for channel in ('prob','logit'):
                original=C.load(model,rep,channel)
                for label,pca,units in [('pca',True,False),('units',False,True),('pca_units',True,True)]:
                    out=C.result_dir('baseline','robustness_'+a.tag,model,rep,channel+'_'+label)
                    if (out/'COMPLETE.json').exists(): continue
                    c=variant(original,pca,units); mm=C.moments(c['Htx'],c['Htt'])
                    obj=C.Objective(c['Vtrain'],lambda A:C.terms(A,mm))
                    C.run_methods(c,out,{'frozen_rule':obj},a,dict(variant=label,PCA_matrix=c.get('PCA_matrix'),
                                  X_unit_factors=c.get('X_unit_factors'),evidence='decoder-autograd',
                                  note='PCA retains all d axes; higher-dimensional PCA/residual protocol not included'))
    for beta in (.00001,.0001):
        for channel in ('prob','logit'):
            out=C.result_dir('baseline','robustness_'+a.tag,'CR3',1,f'{channel}_beta{beta:g}')
            if (out/'COMPLETE.json').exists(): continue
            c=beta_cache(beta,channel); mm=C.moments(c['Htx'],c['Htt'])
            C.run_methods(c,out,{'frozen_rule':C.Objective(c['Vtrain'],lambda A:C.terms(A,mm))},a,
                          dict(checkpoint=c['checkpoint_path'],checkpoint_sha256=c['checkpoint_sha256'],
                               evidence='decoder-autograd',note='Available beta checkpoints only, one NP repeat'))

if __name__=='__main__':
    p=C.parser(__doc__); p.set_defaults(tag='sensitivity_v1'); main(p.parse_args())
