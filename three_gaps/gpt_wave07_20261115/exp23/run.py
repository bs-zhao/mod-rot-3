"""A/B/C on this independent run's fixed physical smoothing fields."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
import numpy as np

class AverageLoss:
    def __init__(self,parts):
        self.parts=parts;self.d=parts[0].d;self.scales=np.stack([p.scales for p in parts])
    def total(self,R):return sum(p.total(R) for p in self.parts)/len(self.parts)

def main(a):
    st=c.Setting(a)
    p=c.OUT/'exp22'/st.key/a.tag/a.outer/'moments.npz'
    if not p.exists():raise FileNotFoundError('own smoothing field must complete first: '+str(p))
    n=a.inner_nodes;scale=[.1,.25,.5]
    with np.load(p) as z:
        parts=[(z[f's{s:g}_n{n}_SX_s'].mean(0),z[f's{s:g}_n{n}_T4_s'].mean(0)) for s in scale]
        A=(z[f'mix_n{n}_SX'].mean(0),z[f'mix_n{n}_T4'].mean(0))
    B=(np.mean([s for s,t in parts],0),np.mean([t for s,t in parts],0))
    for name,pair,objective in [('A_signed',A,None),('B_energy',B,None),('C_loss',B,AverageLoss([c.Loss(s,t,st.C) for s,t in parts]))]:
        c.run_loss(st,f'{a.outer}_n{n}_{name}',*pair,experiment='exp23',objective=objective,
            extra=dict(aggregation=name,scales=scale,weights=[1/3]*3,field_source=str(p),field_sha256=c.sha(p),inner=n,outer=a.outer))

if __name__=='__main__':
    p=c.args(__doc__);p.add_argument('--outer',choices=['pilot','full'],default='pilot');p.add_argument('--inner-nodes',type=int,default=128)
    main(p.parse_args())
