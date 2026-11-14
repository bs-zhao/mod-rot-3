import sys, numpy as np, torch
sys.path.insert(0,r"D:\Projects\rotation\mod-rot-3\three_gaps\exp13_pure_simplicity"); sys.path.insert(0,r"D:\Projects\rotation\mod-rot-3\three_gaps\exp11_cr3_search"); sys.path.insert(0,r"D:\Projects\rotation\mod-rot-3\three_gaps")
import harness as H, screen as SC
from common.rotlib import effective_A, haar
EPS=1e-12
def stat(e,kind):
    e=e.flatten(); q=e/(e.sum()+EPS)
    if kind=="soft": return (e/(e+0.01*e.mean()+EPS)).sum()
    if kind=="neff": return torch.log((-(q+EPS)*(q+EPS).log()).sum().exp())
def collog(e,delta):
    col=e.sum(0); q=e/(col+EPS); w=col/(col.sum()+EPS); return (w*torch.log(q+delta).sum(0)).sum()
for src in ["logit","prob","plugin"]:
    c,cond=SC.load("CR3",1,src); d=4
    ids=H.pure_cells(c["X"].numpy()); g=c["g"].double()
    S2=torch.stack([torch.einsum("ska,skb->ab",g[:,ids==k],g[:,ids==k])/(g.shape[0]*int((ids==k).sum())) for k in range(ids.max()+1)])
    lam=torch.stack([torch.linalg.eigvalsh(e)[0] for e in S2]); print(f"\n{src}: per-cell lambda_min {lam.numpy().round(5)}  tr {S2.diagonal(dim1=1,dim2=2).sum(1).numpy().round(4)}")
    t_dev=torch.tensor(c["V"][c["ids"]],dtype=torch.float32)
    R=np.load(rf"D:\Projects\rotation\mod-rot-3\outputs\three_gaps\exp00\CR3_{cond}_pca0\R_oracle_raw.npy")
    rng=np.random.default_rng(1); Hs=[effective_A(torch.tensor(haar(d,rng),dtype=torch.float32),t_dev) for _ in range(150)]; Ao=effective_A(torch.tensor(R,dtype=torch.float32),t_dev)
    for frac in [0,1,2,3,5,8]:
        S2f=[]
        for e in S2:
            ev,U=torch.linalg.eigh(e); S2f.append((U*(ev-frac*lam.min()).clamp_min(0))@U.T)
        S2f=torch.stack(S2f).float()
        res=[]
        for nm,fn in [("soft",lambda e:stat(e,"soft")),("neff",lambda e:stat(e,"neff")),("colLog3",lambda e:collog(e,1e-3))]:
            v=lambda A: float(fn(torch.einsum("ia,cab,ib->ic",A,S2f,A).clamp_min(0)))
            hv=np.array([v(A) for A in Hs]); res.append((hv.mean()-v(Ao))/(hv.std()+1e-12))
        print(f"  floor x{frac}: z soft {res[0]:.2f} neff {res[1]:.2f} colLog3 {res[2]:.2f}")
