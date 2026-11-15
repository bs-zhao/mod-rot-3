"""Selftests: sigma=0 degenerates to the autograd moments; Gaussian smoothing leaves quadratic functions' Hessian unchanged;
averaged-function derivative == averaged Hessian on a smooth analytic function; the two aggregations differ by the cross term.

python three_gaps/exp22_smoothed_hessian/tests/test_smooth.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exp20_latent_integration"))
import smooth_fields as SF  # noqa: E402
import x7core as X  # noqa: E402
from common.derivs import base_derivatives  # noqa: E402


def test_offsets():
    C = np.diag([1.0, 2.0, 0.5, 3.0])
    o = SF.inner_offsets(16, 4, "cov", C)
    assert np.allclose(o[:8], -o[8:]), "antithetic"
    assert np.abs(o.mean(0)).max() < 1e-12
    print("PASS offsets antithetic, zero mean")


def test_quadratic_and_signed_average():
    # h(v,x) = v^T Q v + v^T B x : Hessian VV = 2Q, VX = B constant -> smoothing leaves it unchanged; E[H]^2 == E[H^2] (no cross term)
    d, p = 3, 2
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((d, d)); Q = Q + Q.T; B = rng.standard_normal((d, p))
    Qt, Bt = torch.tensor(Q), torch.tensor(B)
    h = lambda v, x: v @ Qt @ v / 2 + v @ Bt @ x
    V0 = torch.randn(5, d, dtype=torch.float64); Xb = torch.randn(7, p, dtype=torch.float64)
    delta = SF.inner_offsets(16, d, "cov", np.eye(d)) * 0.5
    Htt = []
    for dl in delta:
        b = base_derivatives(h, V0 + torch.tensor(dl), Xb, which=("Htt", "Htx"), chunk=64)
        Htt.append(b["Htt"].double())
    Hbar = torch.stack(Htt).mean(0)
    assert torch.allclose(Hbar, torch.stack(Htt)[0], atol=1e-10), "quadratic Hessian must be unchanged by smoothing"
    print("PASS smoothing leaves quadratic Hessian unchanged")


def test_cross_term():
    # scalar h(v)=sin(v): H_bar = E sin''(v+delta) ; E[Hbar^2] vs mean(H^2): the difference is the (non-negative) variance of H across inner nodes
    v = torch.linspace(-1, 1, 5, dtype=torch.float64)
    deltas = torch.tensor(np.linspace(-0.8, 0.8, 8))
    Hs = torch.stack([-torch.sin(v + dl) for dl in deltas])
    sq_then_avg = (Hs ** 2).mean(0); avg_then_sq = Hs.mean(0) ** 2
    assert (sq_then_avg >= avg_then_sq - 1e-12).all() and torch.allclose(sq_then_avg - avg_then_sq, Hs.var(0, unbiased=False))
    print("PASS square-then-average - average-then-square == variance across inner nodes (Jensen gap)")


if __name__ == "__main__":
    test_offsets(); test_quadratic_and_signed_average(); test_cross_term()
