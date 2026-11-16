"""Step 2e - how much second-order information do the binary choices contain?  Singular spectrum of the per-item coefficient matrix
(cubic polynomial in U, orthonormal basis w.r.t. the empirical latent measure, 34 non-constant functions x 500 items), for the
noise-free true probabilities and for the observed choices; error of the binary fit after projection on its own top-r subspace.
Mixed true coordinates (diagnostic).  Run from the repository root:  python cld-crossfit-v3/step2_gap12/lowrank.py"""
import sys, numpy as np, torch
sys.path.insert(0, 'cld-crossfit-v3/common'); import xfit as XF
from xfit import X, io
torch.set_num_threads(4)
data, truth = io.load_all("CR3", *XF.BETA["CR3"], 0, 1, need_checkpoint=False)
tr = data["train_ids"]; Tst = truth.std(); K = 500
Q0 = XF.haar(4, np.random.default_rng(20261115)); W = Tst @ Q0.T
Y = data["y"].reshape(-1, K).astype(np.float64); P = data["predictive_true"].reshape(-1, K)
E = XF.exponents(4, 3)
F = XF.mono_feats(torch.tensor(W[tr]), E); Q, Rm = XF._orthobasis(F); n = len(tr)
cP = (Q.T @ torch.tensor(P[tr])) / n; cY = (Q.T @ torch.tensor(Y[tr])) / n          # (35, 500) coefficients in the orthonormal basis
deg = E.sum(1).numpy()
for nm, c in (("noise-free", cP), ("binary", cY)):
    s = torch.linalg.svdvals(c[1:]); e = (s ** 2 / (s ** 2).sum()).numpy()
    print(nm, "singular energy shares:", np.round(e[:14], 4), " cum", np.round(np.cumsum(e)[[3, 5, 7, 9, 11]], 4))
err = cY - cP
print("noise energy / signal energy per polynomial degree block (Q basis is not graded exactly, indicative):")
U, S, Vt = torch.linalg.svd(cP[1:], full_matrices=False)
for r in (4, 6, 8, 10, 12, 16, 34):
    Pr = U[:, :r] @ U[:, :r].T                         # oracle subspace from the noise-free fit (diagnostic)
    Ub, Sb, _ = torch.linalg.svd(cY[1:], full_matrices=False); Pb = Ub[:, :r] @ Ub[:, :r].T   # subspace from the binary fit itself
    sig = (cP[1:] ** 2).sum()
    print(f" r={r:2d}: truncation loss {float(((cP[1:] - Pr @ cP[1:]) ** 2).sum() / sig):.4f} | binary fit projected on its own top-r subspace: "
          f"|err|^2/|signal|^2 = {float(((Pb @ cY[1:] - cP[1:]) ** 2).sum() / sig):.3f}  (unprojected {float((err[1:] ** 2).sum() / sig):.3f})")
# direct split by polynomial degree: the orthonormal basis is built by Gram-Schmidt in degree order, so its columns 1-4 span the linear
# functions of U, 5-14 add the quadratic part, 15-34 the cubic part
c = cP[1:]; tot = float((c ** 2).sum())
print("noise-free energy share by degree: first order %.4f  second order %.4f  third order %.4f" %
      (float((c[:4] ** 2).sum()) / tot, float((c[4:14] ** 2).sum()) / tot, float((c[14:] ** 2).sum()) / tot))
print("share of the top-4 singular functions lying in the linear block: %.4f" % float((U[:4, :4] ** 2).sum() / 4))
nz = err[1:]
print("binary-fit noise energy / noise-free signal energy by degree: first %.2f  second %.2f  third %.2f" %
      tuple(float((nz[a:b] ** 2).sum() / (c[a:b] ** 2).sum()) for a, b in ((0, 4), (4, 14), (14, 34))))
