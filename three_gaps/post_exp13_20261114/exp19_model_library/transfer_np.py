"""Apply the frozen synthetic score to original NPs, without refitting on T."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as C
import run as L

class Score:
    def __init__(self,c,net,mu,sd):
        self.cov=C.cov(c['Vtrain']); self.mm=C.moments(c['Htx'],c['Htt'])
        g=C.tensor(c['g']).reshape(-1,c['Vtrain'].shape[1]); self.s2=g.T@g/len(g)
        self.net,self.mu,self.sd=net,mu,sd
        self.scales=sd; self.weights=torch.ones_like(sd); self.active=torch.ones_like(sd,dtype=torch.bool)
        self.null=self.raw(C.rotations(c['Vtrain'].shape[1],200,1))
    def raw(self,R): return L.features(C.effective(R,self.cov),self.mm,self.s2)
    def __call__(self,R): return self.net((self.raw(R)-self.mu)/self.sd).squeeze(-1)

def main(a):
    checkpoint=C.ROOT/'outputs/three_gaps/post_exp13_20261114/learned_score'/f'{a.frozen_tag}.pt'
    state=torch.load(checkpoint,weights_only=False); sha=C.io.sha256(checkpoint)
    net=torch.nn.Sequential(torch.nn.Linear(6,24),torch.nn.Tanh(),torch.nn.Linear(24,24),torch.nn.Tanh(),torch.nn.Linear(24,1)).double()
    net.load_state_dict(state['state']); net.eval(); net.requires_grad_(False)
    for model in C.COND:
        for rep in (1,2):
            for channel in ('prob','logit'):
                out=C.result_dir('exp19_model_library',a.tag,model,rep,channel)
                if (out/'COMPLETE.json').exists(): continue
                c=C.load(model,rep,channel)
                obj=Score(c,net,state['mu'],state['sd'])
                C.run_methods(c,out,{'frozen_meta_score':obj},a,
                              dict(evidence='decoder-autograd; frozen synthetic meta-score transfer',
                                   frozen_checkpoint_sha256=sha,meta_model_refit=False,
                                   note='Scales in rotations.npz are training feature SD, not Haar weights; objective is a nonlinear MLP',
                                   limitation='Synthetic train d=4, X=4; C3 d=6 and CR3 X=6 are out-of-distribution'))
    if C.io.sha256(checkpoint)!=sha: raise RuntimeError('Frozen score changed')

if __name__=='__main__':
    p=C.parser(__doc__); p.set_defaults(tag='np_transfer_v1'); p.add_argument('--frozen-tag',default='grammar_v1')
    main(p.parse_args())
