import unittest
import numpy as np
import torch
from core import formula
from transport import operator_targets,flow_input
from metrics import Loss

class EquivarianceTests(unittest.TestCase):
    def test_flow_generators_include_zero_amount_trials(self):
        x=np.array([[100,0,.5,0,0,.5],[50,0,.3,100,0,.7],[0,100,.5,0,50,.5],[80,20,.2,100,50,.8]],float)
        t=torch.tensor([.8,2.,.7,2.2],dtype=torch.float64)
        w=torch.cat((t.expand(len(x),-1),torch.tensor(x)),dim=1)
        def fn(w):return formula('CR3',w[:4],w[4:])
        g=torch.vmap(torch.func.jacrev(fn))(w).numpy()[:,:4]
        coefficients=np.array([.8,.8*2,.7,2.2])
        for kind in range(3):
            def along(e):return torch.vmap(fn)(flow_input(w,e,kind,4,torch.zeros(6,dtype=torch.float64),torch.ones(6,dtype=torch.float64)))
            _,got=torch.func.jvp(along,(torch.tensor(0.,dtype=torch.float64),),(torch.tensor(1.,dtype=torch.float64),))
            np.testing.assert_allclose(got.numpy(),g[:,kind]*coefficients[kind],atol=1e-12,rtol=1e-12)

    def test_cr3_input_generator_identities_and_true_axis_minimum(self):
        rng=np.random.default_rng(81);n=3;k=32
        x=rng.uniform(5,200,(k,6));x[:,[2,5]]=rng.uniform(.05,.95,(k,2))
        t=np.array([[.8,2.,.7,2.2],[.6,1.4,.9,1.8],[.9,2.8,.6,2.5]])
        w=np.concatenate((np.repeat(t,k,axis=0),np.tile(x,(n,1))),axis=1)
        def fn(w):return formula('CR3',w[:4],w[4:])
        wt=torch.tensor(w);value=torch.vmap(fn)(wt).numpy().reshape(n,k)
        gradient=torch.vmap(torch.func.jacrev(fn))(wt).numpy().reshape(n,k,10)
        targets=operator_targets(x,gradient[...,4:],value)
        coefficients=np.column_stack((t[:,0],t[:,0]*t[:,1],t[:,2],t[:,3]))
        np.testing.assert_allclose(targets,gradient[...,:4]*coefficients[:,None,:],atol=1e-12,rtol=1e-12)
        gz=gradient[...,:4];weights=np.ones((n,k))/k
        fields=dict(g=gz.reshape(-1,4),transport_gram=np.einsum('ns,nsa,nsb->nab',weights,gz,gz),
                    transport_cross=np.einsum('ns,nsa,nsk->nak',weights,gz,targets),
                    transport_energy=np.einsum('ns,nsk,nsk->nk',weights,targets,targets),
                    transport_vector_grams=np.zeros((4,4,4)))
        loss=Loss(fields,np.eye(4),4,'transport_profile')
        self.assertLess(abs(loss(np.eye(4))),1e-12)
        q,_=np.linalg.qr(rng.normal(size=(4,4)))
        self.assertGreater(loss(q),.01)
        scaled=np.array([.01,-100,.3,-4.])[:,None]*q
        self.assertAlmostEqual(loss(q),loss(scaled),places=12)

if __name__=='__main__':unittest.main()
