import numpy as np
import torch
from .objectives import sym_basis,sym_power,rotation,ProgramObjective,RecurrenceObjective,optimize
from .jets import directional
from .toys import run_toys
from .toys import analytic_blocks


def test_symmetric_power_preserves_tensor_metric_and_direction():
    a=rotation(torch.tensor([.2,-.3,.1],dtype=torch.float64),3)
    rng=np.random.default_rng(3)
    for degree in [1,2,3]:
        groups,b=sym_basis(3,degree);b=torch.tensor(b,dtype=torch.float64)
        p=sym_power(a,b,degree)
        assert torch.max(torch.abs(p@p.T-torch.eye(len(groups))))<1e-12
        packed=torch.tensor(rng.normal(size=len(groups)),dtype=torch.float64)
        tensor=(b@packed).reshape((3,)*degree)
        row=a[0]
        value=torch.einsum('i,i->',row,tensor) if degree==1 else (torch.einsum('i,j,ij->',row,row,tensor) if degree==2 else torch.einsum('i,j,k,ijk->',row,row,row,tensor))
        assert torch.allclose((p@packed)[groups.index((0,)*degree)],value)


def test_nested_forward_ad_keeps_mixed_and_pure_derivatives():
    v=torch.tensor([[.7,.3,-.2],[1.1,-.4,.5]],dtype=torch.float64)
    fun=lambda t:t[:,0]*torch.exp(t[:,1]*t[:,2])
    expected=torch.exp(v[:,1]*v[:,2])
    assert torch.allclose(directional(fun,v,(0,1,2)),expected*(1+v[:,1]*v[:,2]))
    assert torch.allclose(directional(fun,v,(0,0)),torch.zeros(2,dtype=torch.float64))
    assert torch.allclose(directional(fun,v,(1,1,1)),v[:,0]*v[:,2]**3*expected)


def test_toy_gates_and_linear_recurrence_nonidentifiability():
    results,_=run_toys()
    for toy in results.values():
        for key in ['group_l1','module_code']:
            assert toy[key]['identity']<=toy[key]['min_loss']+1e-10
            assert toy[key]['at_45']>toy[key]['identity']+.1
    assert results['linear_stimulus']['recurrence']['range']==0
    assert results['bilinear']['recurrence']['at_45']>.01


def test_four_dimensional_hidden_rotation_recovers_function_roles():
    from scipy.optimize import linear_sum_assignment
    rng=np.random.default_rng(81)
    hidden=rotation(torch.tensor(rng.normal(0,.5,6)),4)
    def fun(v):
        theta=hidden.T@v[:4]
        return theta[0]+torch.sum(theta[1:]*v[4:])
    q=rng.normal(size=(12,4));x=rng.uniform(-1,1,(64,3))
    blocks=analytic_blocks(fun,q,x);program=ProgramObjective(blocks,4,3)
    a,_,_=optimize(lambda a:program.module_loss(a,'module_code'),4,starts=3)
    score=np.abs(a@hidden.numpy());i,j=linear_sum_assignment(-score)
    assert np.mean(score[i,j])>.999


def test_x_projection_residual_cost_is_rotation_invariant():
    from .functional import solve
    rng=np.random.default_rng(28);n,x,k=8,60,3
    target=rng.normal(size=(x,n,k));phi=np.column_stack((np.ones(x),rng.normal(size=(x,5))))
    ids=np.arange(40);a=rotation(torch.tensor([.2,.3,-.1]),k).numpy()
    first=solve(phi,target.reshape(x,n*k),ids,.001)
    residual=target-(phi@first).reshape(x,n,k)
    rotated=target@a.T
    second=solve(phi,rotated.reshape(x,n*k),ids,.001)
    residual_rot=rotated-(phi@second).reshape(x,n,k)
    assert np.allclose(residual_rot,residual@a.T)
    assert np.isclose(np.sum(residual_rot**2),np.sum(residual**2))


def test_function_symmetries_identify_additive_and_gain_roles():
    from .symmetries import detect,fields
    rng=np.random.default_rng(72);n=128;nx=64
    theta=rng.normal(size=(n,4));x=rng.uniform(-1,1,(nx,3));phi=np.column_stack((np.ones(nx),x))
    eta=theta@phi.T;g=np.broadcast_to(phi,(n,nx,4)).copy()
    frame,count,reports=detect(eta,g)
    assert count==1 and reports[0]['accepted']
    assert abs(frame[0,0])>.999999
    a=rng.uniform(.5,1.5,n);b=rng.uniform(-1.5,1.5,n);logx=np.linspace(-1,1,nx)
    eta=a[:,None]*np.exp(b[:,None]*logx)
    g=np.stack((eta/a[:,None],eta*logx),axis=-1)
    _,_,gain=fields(eta,g);e,v=np.linalg.eigh(gain.mean(0))
    assert e[0]<1e-14 and abs(v[0,0])>.99999


def test_redundant_multiplicative_scale_is_not_unique_axis():
    from .symmetries import detect
    rng=np.random.default_rng(73);n,nx=128,64
    a=rng.uniform(.5,1.5,n);b=rng.uniform(.5,1.5,n);x=rng.normal(size=nx)
    eta=a[:,None]*b[:,None]*x
    g=np.stack((b[:,None]*np.broadcast_to(x,(n,nx)),a[:,None]*np.broadcast_to(x,(n,nx))),axis=-1)
    _,count,reports=detect(eta,g)
    assert count==0
