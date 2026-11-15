"""Selftest: the stencil formulas are exact for quadratic / bilinear functions and agree with autograd for h -> 0 on a smooth function.

python three_gaps/exp21_data_scale_hessian/tests/test_stencils.py
"""
import torch

torch.set_default_dtype(torch.float64)


def central2(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2


def mixed(f, x, y, hx, hy):
    return (f(x + hx, y + hy) - f(x + hx, y - hy) - f(x - hx, y + hy) + f(x - hx, y - hy)) / (4 * hx * hy)


def test_exact_quadratic_bilinear():
    q = lambda x: 3 * x ** 2 + 2 * x + 1
    assert abs(central2(q, torch.tensor(0.7), 0.5) - 6.0) < 1e-10, "axial second difference exact for quadratics for any h"
    b = lambda x, y: 2 * x * y + x ** 2 * 0 + 5 * x
    assert abs(mixed(b, torch.tensor(0.3), torch.tensor(-0.4), 0.7, 0.2) - 2.0) < 1e-10, "mixed difference exact for bilinear"
    print("PASS stencils exact for quadratic / bilinear at any half-width")


def test_limit_and_noise_bound():
    f = lambda x: torch.sin(2 * x)
    x = torch.tensor(0.3, requires_grad=True)
    g = torch.autograd.grad(f(x), x, create_graph=True)[0]
    h2 = torch.autograd.grad(g, x)[0]
    assert abs(central2(f, x.detach(), 1e-3) - h2.item()) < 1e-4, "h -> 0 limit equals autograd"
    eps, w, h = 0.01, 20.0, 0.1
    fn = lambda t: t ** 2 + eps * torch.sin(w * t)
    err = abs(central2(fn, torch.tensor(0.2), h) - 2.0)
    assert err <= 4 * eps / h ** 2 + 1e-9, "error bound 4 eps / h^2"
    print("PASS h->0 equals autograd; high-frequency perturbation error within 4*eps/h^2")


if __name__ == "__main__":
    test_exact_quadratic_bilinear(); test_limit_and_noise_bound()
