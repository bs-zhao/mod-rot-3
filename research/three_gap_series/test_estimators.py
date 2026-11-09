import unittest
import numpy as np
import torch
from core import derivatives
from estimators import PolynomialSieve,BankSieve

class SieveTests(unittest.TestCase):
    def test_bank_derivatives_match_selected_stimulus(self):
        rng=np.random.default_rng(41);w=rng.normal(size=(300,2))
        y=np.column_stack((w[:,0]*w[:,1],w[:,0]**2+.3*w[:,1]))
        sieve=BankSieve(2,0,'logit').fit(w,y,w[:10],y[:10])
        sieve.eval_stimulus=np.array([0,1,0,1])
        got=sieve.fields(w[:4])
        target=np.array([[0.,1.],[2.,0.],[0.,1.],[2.,0.]])
        np.testing.assert_allclose(got['h'][:,0,:],target,atol=1e-10)
        averaged=sieve.mean_fields(w)
        np.testing.assert_allclose(averaged['g'],[[w[:,1].mean(),w[:,0].mean()],[2*w[:,0].mean(),.3]],atol=1e-10)
    def test_polynomial_and_sigmoid_third_derivatives(self):
        rng=np.random.default_rng(71)
        w=rng.normal(size=(1200,3));ev=rng.normal(size=(12,3))
        def fn(x): return .3+x[0]*x[1]**2-.4*x[0]*x[2]+.2*x[1]
        with torch.no_grad(): y=torch.vmap(fn)(torch.tensor(w)).numpy()
        truth=derivatives(fn,ev,3,True)
        for channel in ['logit','probability']:
            yy=y if channel=='logit' else 1/(1+np.exp(-y))
            sieve=PolynomialSieve(3,0,channel).fit(w,yy,w[:20],yy[:20])
            got=sieve.fields(ev,True)
            target=truth if channel=='logit' else derivatives(lambda x:torch.sigmoid(fn(x)),ev,3,True)
            for key in got: np.testing.assert_allclose(got[key],target[key],atol=1e-10)

    def test_ridge_fit_is_rotation_equivariant(self):
        rng=np.random.default_rng(9)
        w=rng.normal(size=(400,4));ev=rng.normal(size=(9,4));y=np.sin(w[:,0])+w[:,1]*w[:,2]
        q,_=np.linalg.qr(rng.normal(size=(4,4)))
        first=PolynomialSieve(3,.01).fit(w,y,w[:10],y[:10]).fields(ev)
        second=PolynomialSieve(3,.01).fit(w@q.T,y,w[:10]@q.T,y[:10]).fields(ev@q.T)
        np.testing.assert_allclose(first['g'],second['g']@q,atol=1e-10)
        np.testing.assert_allclose(first['h'],np.einsum('ai,nab,bj->nij',q,second['h'],q),atol=1e-10)

if __name__=='__main__': unittest.main()
