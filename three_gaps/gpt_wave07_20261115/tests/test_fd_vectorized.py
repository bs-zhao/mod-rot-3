"""Actual same-node FD energy/gradient equivalence; no T or Oracle access."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'exp21'))
import core as c
from stencils import Field,FDLoss
from scales import resolution
import torch
import numpy as np

def main(a):
    st=c.Setting(a)
    K=64 if st.dev.type=='cuda' else 8;S=64 if st.dev.type=='cuda' else 4
    xi=np.sort(np.random.default_rng(21).choice(len(st.X),K,False));V=st.V[st.ids0[:S]];X=st.X[xi]
    gap=resolution(st.X)['gap_quantiles'][:,1]
    dtype=torch.float32 if st.dev.type=='cuda' else torch.float64
    f=[Field(st,V,X,dtype,engine=e) for e in ['scalar','vectorized']]
    rows=[]
    for mode in ['hybrid','ambient']:
        for h in [.05,.25,.5]:
            results=[]
            for field in f:
                R=torch.tensor(c.haar_set(st.d,1,23)[0],dtype=torch.float32,requires_grad=True)
                if st.dev.type=='cuda':torch.cuda.synchronize()
                t=time.perf_counter();ux,uu=field.energies(R,h,mode,gap)
                q=ux/(ux.sum()+1e-12)+1e-12
                L=(-(q*q.log()).sum()).exp()+uu.diag().sum()/(uu.sum()+1e-12)
                L.backward()
                if st.dev.type=='cuda':torch.cuda.synchronize()
                results.append((ux.detach().cpu().numpy(),uu.detach().cpu().numpy(),R.grad.numpy(),time.perf_counter()-t))
            x,y=results
            err=[float(np.linalg.norm(x[i]-y[i])/max(np.linalg.norm(x[i]),1e-12)) for i in range(3)]
            if max(err[:2])>.005 or err[2]>.01:raise AssertionError((mode,h,err))
            rows.append(dict(mode=mode,h=h,UX_relative=err[0],UU_relative=err[1],gradient_relative=err[2],scalar_seconds=x[3],vectorized_seconds=y[3]))
    c.dump(c.HERE/'tests'/f'fd_vectorized_{st.dev.type}.json',dict(passed=True,device=str(st.dev),subjects=S,X_points=K,rows=rows,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if st.dev.type=='cuda' else None))
    print('ACTUAL FD VECTORIZED ENERGY/GRADIENT TEST PASS',rows)

if __name__=='__main__':main(c.args(__doc__).parse_args())
