"""Independent completion of the EXP13 tuning grid; no historical files touched.

All candidates are reported. These benchmark TEST sets were already used in
development, so the table is exploratory, not an independent model-selection test.
"""
import itertools
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as C

def main(a):
    c=C.load(a.model,a.repeat,a.channel,300,500)
    out=C.result_dir('baseline','tuning_'+a.tag,a.model,a.repeat,a.channel)
    mats={s:C.moments(c['Htx']*(c['X'].std(0) if s else 1),c['Htt']) for s in (0,1)}
    def stats(e,name):
        if name=='neff': return C.neff(e)
        if name=='l1l2': return e.clamp_min(0).sqrt().sum((-2,-1))/(e.sum((-2,-1)).sqrt()+C.EPS)
        if name=='quartimax': return -(e/(e.sum((-2,-1),keepdim=True)+C.EPS)).square().sum((-2,-1))
        if name=='softL0': return (e/(e+.01*e.mean((-2,-1),keepdim=True)+C.EPS)).sum((-2,-1))
    methods={}
    for stat,curv,w,s in itertools.product(('neff','l1l2','quartimax','softL0'),('share','sum'),(0.,.25,.5,1.,2.,4.),(0,1)):
        if w==0 and curv=='sum': continue # identical zero-weight duplicate
        def f(A,stat=stat,curv=curv,s=s):
            ux,uu=C.energies(A,mats[s]); u=stats(ux,stat)
            v=C.share(uu) if curv=='share' else uu.diagonal(dim1=-2,dim2=-1).sum(-1)
            return torch.stack([u,v],-1)
        key=f'{stat}_{curv}_w{w:g}_stdX{s}'
        methods[key]=C.Objective(c['Vtrain'],f,[1,w])
    if a.quick:
        methods={k:v for k,v in methods.items() if k.startswith('neff_share')}
    C.run_methods(c,out,methods,a,dict(development=True,selection='none; publish complete candidate grid',
                   independent_repeats_3_to_5='not available; require additional NP training'))

if __name__=='__main__':
    p=C.parser(__doc__); p.add_argument('--quick',action='store_true'); main(p.parse_args())
