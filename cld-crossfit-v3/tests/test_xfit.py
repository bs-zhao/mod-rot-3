"""Unit tests for common/xfit.py.   python -m pytest cld-crossfit-v3/tests -q     (or run the file directly)"""
import sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import xfit as XF
from xfit import X as X6                     # frozen EXP13 pieces (losses2 as X6.L2)
torch.manual_seed(0)
D = torch.float64


def _haar(d, seed):
    return torch.tensor(XF.haar(d, np.random.default_rng(seed)), dtype=D)


def test_monomial_derivatives_match_autograd():
    d, du = 4, 3; E = XF.exponents(d, du); v = torch.randn(7, d, dtype=D)
    F, dF, d2F = XF.mono_feats(v, E, order=2)
    f = lambda x: XF.mono_feats(x[None], E)[0]
    for i in range(len(v)):
        J = torch.autograd.functional.jacobian(f, v[i])                       # (m,d)
        assert torch.allclose(J, dF[i], atol=1e-10)
        for a in range(E.shape[0]):
            Hm = torch.autograd.functional.hessian(lambda x, a=a: f(x)[a], v[i])
            assert torch.allclose(Hm, d2F[i, a], atol=1e-9)


def _toy(n=4000, K=30, d=3, seed=1):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(n, d, generator=g, dtype=D) * torch.tensor([1.5, 1.0, 0.7], dtype=D)
    b = torch.randn(K, d, generator=g, dtype=D); C = torch.randn(K, d, d, generator=g, dtype=D) * 0.3; C = C + C.transpose(1, 2)
    eta = v @ b.T + 0.5 * torch.einsum("na,kab,nb->nk", v, C, v)
    return v, eta, b, C


def test_fits_are_equivariant_under_linear_maps():
    """Fit in coordinates w = v B^T and transform the fields back  ==  fit directly in v."""
    v, eta, _, _ = _toy(); Y = (torch.rand(eta.shape, dtype=D, generator=torch.Generator().manual_seed(2)) < torch.sigmoid(eta)).to(D)
    Bm = _haar(3, 5) @ torch.diag(torch.tensor([1.0, 2.0, 0.5], dtype=D)) @ _haar(3, 6)
    w = v @ Bm.T; E = XF.exponents(3, 2); grid = v[:50]
    for link, fit in (("identity", XF.fit_lpm), ("logit", XF.fit_logistic)):
        c1 = fit(XF.mono_feats(v, E), Y); c2 = fit(XF.mono_feats(w, E), Y)
        f1 = XF.poly_fields(c1, grid, E, link)["prob"]; f2 = XF.poly_fields(c2, grid @ Bm.T, E, link)["prob"]
        assert torch.allclose(f1["h"], f2["h"], atol=1e-6)
        assert torch.allclose(f1["g"], torch.einsum("ska,ab->skb", f2["g"], Bm), atol=1e-6)            # d/dv = B^T d/dw
        assert torch.allclose(f1["Htt"], torch.einsum("skab,ac,bd->skcd", f2["Htt"], Bm, Bm), atol=1e-6)


def test_logistic_recovers_quadratic_logit():
    v, eta, b, C = _toy(n=60000); E = XF.exponents(3, 2)
    c = XF.fit_logistic(XF.mono_feats(v, E), torch.sigmoid(eta), ridge=0.0)                              # noise-free soft targets
    f = XF.poly_fields(c, v[:20], E, "logit")["logit"]
    assert torch.allclose(f["Htt"], C[None].expand(20, -1, -1, -1), atol=1e-5)
    assert torch.allclose(f["g"], b[None] + torch.einsum("kab,sb->ska", C, v[:20]), atol=1e-5)


def test_pair_formula_equals_explicit_loop():
    ts = [torch.randn(5, 6, 3, 3, dtype=D) for _ in range(4)]
    ref = sum(torch.einsum("skab,skcd->abcd", ts[a], ts[b]) for a in range(4) for b in range(4) if a != b) / 12
    assert torch.allclose(XF._pairs(ts, "skab,skcd->abcd"), ref, atol=1e-10)


def test_cross_moments_have_no_noise_floor():
    """signal with exact zeros + independent noise per fold: plain energies get +sigma^2 everywhere, cross-fitted ones do not."""
    S, K, d, sig = 400, 60, 4, 0.5
    H0 = torch.zeros(S, K, d, d, dtype=D); H0[..., 0, 1] = H0[..., 1, 0] = 1.0                         # only the (0,1) interaction is real
    folds = [dict(g=torch.zeros(S, K, d, dtype=D), Htt=H0 + sig * torch.randn(S, K, d, d, dtype=D)) for _ in range(2)]
    A = torch.eye(d, dtype=D)
    ec = XF.e_uu(A, XF.moments(folds, cross=True)); ep = XF.e_uu(A, XF.moments(folds, cross=False))
    assert abs(float(ec[0, 1]) - 1.0) < 0.02 and float(ec[2, 3].abs()) < 0.01 and float(ec[2, 2].abs()) < 0.01
    assert abs(float(ep[2, 3]) - sig ** 2) < 0.02 and abs(float(ep[0, 1]) - (1 + sig ** 2)) < 0.03


def test_energies_match_frozen_exp13_when_folds_are_identical():
    S, K, d, p = 20, 15, 4, 6
    c = dict(g=torch.randn(S, K, d), Htt=torch.randn(S, K, d, d), Htx=torch.randn(S, K, d, p))
    SX, T4 = X6.L2.moments(dict(c))
    m = XF.moments([c], cross=False)
    A = torch.randn(d, d)
    ux, uu = X6.L2.energies(A, SX, T4)
    assert torch.allclose(XF.e_ux(A.double(), m).float(), ux, atol=1e-4) and torch.allclose(XF.e_uu(A.double(), m).float(), uu, atol=1e-3)
    t = XF.make_terms(m)
    assert abs(float(t["UX_neff"](A.double())) - float(X6.L2.LOSSES["UX_neff"](ux, uu))) < 1e-3
    assert abs(float(t["UUd_share"](A.double())) - float(X6.L2.LOSSES["UUd_share"](ux, uu))) < 1e-4


def test_search_recovers_axes_of_a_sparse_quadratic():
    """h = a*x1 + b*x2 + c*d*x3 in rotated coordinates: UUd_share + item-level sparsity must find the signed permutation."""
    d, S, K = 4, 300, 40; g0 = torch.Generator().manual_seed(3)
    t = torch.randn(S, d, generator=g0, dtype=D); x = torch.randn(K, 3, generator=g0, dtype=D)
    x[: K // 2, 1] = 0; x[K // 2:, 0] = 0                                        # a and b act on disjoint sets of stimuli (else dense Gaussian loadings are rotation invariant)
    H = torch.zeros(S, K, d, d, dtype=D); H[:, :, 2, 3] = H[:, :, 3, 2] = x[None, :, 2]
    g = torch.zeros(S, K, d, dtype=D); g[..., 0] = x[None, :, 0]; g[..., 1] = x[None, :, 1]
    g[..., 2] = t[:, None, 3] * x[None, :, 2]; g[..., 3] = t[:, None, 2] * x[None, :, 2]
    Q = _haar(d, 9); w = t @ Q.T                                                 # w = Q t  ->  d/dw = Q d/dt
    f = dict(g=torch.einsum("ska,ba->skb", g, Q), Htt=torch.einsum("skab,ca,db->skcd", H, Q, Q))
    m = XF.moments([f], cross=False); terms = XF.make_terms(m); sc, _ = XF.haar_scales(terms, w)
    fin = XF.search(XF.combo(terms, sc, ("UI_neff", "UUd_share")), w, starts=8, steps=200)
    assert XF.axis_error(fin[0][1] @ Q.numpy())[0] < 3.0


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
