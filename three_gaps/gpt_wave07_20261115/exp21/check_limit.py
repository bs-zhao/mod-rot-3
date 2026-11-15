"""Actual decoder FD -> autograd limit, original float32 vs float64 copy."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import core as c
from stencils import Field
import numpy as np
import torch

def main(a):
    if a.channel!='prob':raise ValueError('probability limit test')
    st=c.Setting(a);V=st.V[st.ids0[:4]];X=st.X[:8];field=Field(st,V,X,torch.float64)
    R=torch.tensor(c.haar_set(st.d,1,13)[0],dtype=torch.float64)
    with torch.no_grad():
        u0,t0=field.energies(R,0,'autograd')
        rows=[]
        for h in [.001,.003,.01,.05,.1,.25,.5,1.]:
            u,t=field.energies(R,h,'ambient',h*st.X.std(0))
            rows.append(dict(h_U_halfwidth=h,h_X_halfwidth=(h*st.X.std(0)).tolist(),UX_relative=float(torch.linalg.norm(u-u0)/torch.linalg.norm(u0)),UU_relative=float(torch.linalg.norm(t-t0)/torch.linalg.norm(t0))))
    c.dump(c.HERE/'exp21/results'/st.key/a.tag/'limit_check.json',rows)
    if rows[0]['UX_relative']>.001 or rows[0]['UU_relative']>.001:raise AssertionError('finite difference does not converge to autograd')
    f32=Field(st,V,X,torch.float32);precision=[]
    with torch.no_grad():
        for h in [.001,.003,.01,.05,.1,.25,.5]:
            u,t=f32.energies(R,h,'ambient',h*st.X.std(0));u64,t64=field.energies(R,h,'ambient',h*st.X.std(0))
            precision.append(dict(h=h,UX_relative_32_vs_64=float(torch.linalg.norm(u-u64)/torch.linalg.norm(u64)),UU_relative_32_vs_64=float(torch.linalg.norm(t-t64)/torch.linalg.norm(t64))))
    c.dump(c.HERE/'exp21/results'/st.key/a.tag/'precision_32_vs_64.json',precision)
    print(rows);print(precision)

if __name__=='__main__':main(c.args(__doc__).parse_args())
