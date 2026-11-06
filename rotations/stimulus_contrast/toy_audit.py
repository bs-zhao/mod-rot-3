"""Exact counterexamples and the requested power-function toy.

These are objective diagnostics only, never benchmark fit inputs.
"""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rotations.stimulus_contrast.method import matrices,loss,from_angles


def exact_linear_contrast(a):
    # phi=(1,sqrt(3)*X1,sqrt(3)*X2,sqrt(3)*X3), iid X~U(-1,1).
    # E ||phi phi^T||² = 18.4; E phi phi^T=I.
    # Sum fourth diagonal moments = 12-2 sum_i a[i,0]^4
    #                                  -1.2 sum_{i,j>0} a[i,j]^4.
    diag_fourth=12.-2.*np.sum(a[:,0]**4)-1.2*np.sum(a[:,1:]**4)
    off=18.4-diag_fourth
    return float(off/14.4),float(off/18.4)


def main():
    folder=ROOT/'outputs'/'research'/'stimulus_contrast'/'analytic_v1'
    folder.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(81)
    x=rng.uniform(-np.sqrt(3),np.sqrt(3),(150000,3))
    phi=np.column_stack((np.ones(len(x)),x))
    g=phi.T[None,:,:]
    h=np.array([[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]],float)/2
    c=matrices(g); pooled=matrices(g,False)
    report=dict(linear=dict(identity=dict(exact_contrast=exact_linear_contrast(np.eye(4))[0],
                                         numerical_contrast=loss(np.eye(4),c),
                                         exact_pooled=exact_linear_contrast(np.eye(4))[1],numerical_pooled=loss(np.eye(4),pooled)),
                           mixed_hadamard=dict(exact_contrast=exact_linear_contrast(h)[0],numerical_contrast=loss(h,c),
                                               exact_pooled=exact_linear_contrast(h)[1],numerical_pooled=loss(h,pooled)),
                           gradient_rank_per_axis='one for every rotation: gradient does not depend on subject'))
    # f(theta,x)=theta1*x**theta2. Positive theta1; common independent x.
    theta=np.column_stack((rng.uniform(.5,1.5,512),rng.uniform(-1,1,512)))
    xx=np.exp(np.linspace(-1,1,128))
    basis=xx[None,:]**theta[:,1,None]
    power_g=np.stack((basis,theta[:,0,None]*np.log(xx)[None,:]*basis),axis=1)
    pm=matrices(power_g)
    angles=np.linspace(-90,90,181)
    power_loss=[loss(from_angles([np.deg2rad(v)],2),pm) for v in angles]
    report['power']=dict(theta1_range=[.5,1.5],theta2_range=[-1,1],log_x_range=[-1,1],
                         true_coordinate_loss=loss(np.eye(2),pm),
                         minimum_loss=float(min(power_loss)),minimum_angle=float(angles[int(np.argmin(power_loss))]),
                         interpretation='A minimum away from 0 or +/-90 is wrong parameter mixing, not optimizer failure.')
    (folder/'analytic.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(['True axes','Hadamard mix'],[report['linear']['identity']['exact_contrast'],report['linear']['mixed_hadamard']['exact_contrast']])
    axes[0].set_ylabel('Exact contrast covariance loss'); axes[0].set_title('Perfect linear decoder: mixed axes win')
    axes[1].plot(angles,power_loss);axes[1].axvline(0,c='gray');axes[1].set_xlabel('Rotation angle (degrees)')
    axes[1].set_ylabel('Contrast covariance loss');axes[1].set_title('Power-function diagnostic')
    fig.tight_layout();fig.savefig(folder/'toy_counterexamples.png',dpi=160)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
