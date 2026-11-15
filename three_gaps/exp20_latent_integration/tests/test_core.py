"""Selftests: streaming moments == frozen L2.moments on the same subjects; covariance-based A == frozen effective_A.

python three_gaps/exp20_latent_integration/tests/test_core.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import x7core as X  # noqa: E402
import losses2 as L2  # noqa: E402
from common.rotlib import effective_A, haar  # noqa: E402


def test_streaming_matches_frozen_moments(model="CR3", rep=1, n=6):
    st = X.Setting(model, rep)
    c = torch.load(st.cache_path, weights_only=False)
    ids = c["ids"][:n]
    SX_a, T4_a = X.centre_moments(st, "prob", st.V[ids], bs=3)
    SX_b, T4_b = X.centre_moments(st, "prob", st.V[ids], bs=n)
    assert np.allclose(SX_a, SX_b, rtol=1e-5, atol=1e-9) and np.allclose(T4_a, T4_b, rtol=1e-5, atol=1e-9), "batch-boundary dependence"
    cc = dict(c)
    for k in ("g", "Htt", "Htx", "h"):
        cc[k] = c[k][:n]
    cc["ids"] = ids
    SXf, T4f = L2.moments(cc)
    assert np.allclose(SX_a.mean(0), SXf.numpy(), rtol=2e-3, atol=1e-8), "SX mismatch vs frozen moments"
    assert np.allclose(T4_a.mean(0), T4f.numpy(), rtol=2e-3, atol=1e-8), "T4 mismatch vs frozen moments"
    print("PASS streaming == frozen moments (tolerance 2e-3: the frozen cache is float32), batch-boundary invariant")


def test_loss_equivalence(model="CR3", rep=1):
    st = X.Setting(model, rep)
    c = torch.load(st.cache_path, weights_only=False)
    SX, T4 = L2.moments(c)
    Cref = X.population_cov(st.V[c["ids"]])
    ml = X.MomentLoss(SX.numpy(), T4.numpy(), Cref, st.d)
    t_dev = torch.tensor(st.V[c["ids"]], dtype=torch.float32)
    M = torch.tensor(haar(st.d, np.random.default_rng(5)), dtype=torch.float32)
    e1 = L2.energies(effective_A(M, t_dev), SX, T4)
    e2 = ml.energies(M)
    assert all(torch.allclose(a, b, rtol=1e-3, atol=1e-9) for a, b in zip(e1, e2)), "frozen effective_A vs covariance A differ"
    print("PASS eff_A(cov) == frozen effective_A; energies identical")


if __name__ == "__main__":
    test_streaming_matches_frozen_moments()
    test_loss_equivalence()
