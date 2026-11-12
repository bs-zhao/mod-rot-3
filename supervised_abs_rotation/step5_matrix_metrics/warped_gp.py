"""Global monotone output calibration using a low-rank GP change-of-variables density.

The piecewise-linear link has positive slopes. Its Jacobian is included;
the objective cannot improve merely by collapsing all transformed outputs.
All linear-algebra inputs are sufficient statistics of actual W/Y rows.
"""
import numpy as np
import torch


def hat_basis(y, knots):
    y, knots = np.asarray(y), np.asarray(knots)
    bins = np.clip(np.searchsorted(knots, y, side='right')-1, 0, len(knots)-2)
    weight = (y-knots[bins])/(knots[bins+1]-knots[bins])
    basis = np.zeros((len(y), len(knots)))
    basis[np.arange(len(y)), bins] = 1-weight
    basis[np.arange(len(y)), bins+1] = weight
    return basis, bins


def inverse(values, knots, heights):
    bins = np.clip(np.searchsorted(heights, values, side='right')-1, 0, len(heights)-2)
    return knots[bins]+(values-heights[bins])*(knots[bins+1]-knots[bins])/(heights[bins+1]-heights[bins])


class Model(torch.nn.Module):
    def __init__(self, gram, cross, response_gram, response_sum, bin_counts, knots, learn_warp=True):
        super().__init__()
        tensor = lambda a: torch.as_tensor(a, dtype=torch.float64)
        values, axes = torch.linalg.eigh(tensor((gram+gram.T)/2))
        if float(values.min()) < -1e-8*max(float(values.max()), 1.):
            raise ArithmeticError('Actual-row GP feature Gram is not positive semidefinite')
        self.values, self.axes = torch.clamp(values, min=0.), axes
        self.cross = tensor(cross); self.eigen_cross = axes.T @ self.cross
        self.response_gram, self.response_sum = tensor(response_gram), tensor(response_sum)
        self.counts, self.knots = tensor(bin_counts), tensor(knots)
        self.n = float(np.sum(bin_counts)); self.learn_warp = learn_warp
        self.log_slopes = torch.nn.Parameter(torch.zeros(len(knots)-1, dtype=torch.float64), requires_grad=learn_warp)
        self.log_amplitude = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.log_noise = torch.nn.Parameter(torch.tensor(np.log(.1), dtype=torch.float64))

    def heights(self):
        slopes = torch.exp(self.log_slopes-self.log_slopes.mean())
        values = torch.cat((torch.zeros(1, dtype=torch.float64), torch.cumsum(torch.diff(self.knots)*slopes, dim=0)))
        center = self.response_sum @ values/self.n
        variance = values @ self.response_gram @ values/self.n-center.square()
        if float(variance.detach()) <= 0.:
            raise ArithmeticError('Warp normalization has zero variance')
        return (values-center)/torch.sqrt(variance)

    def nll(self):
        heights = self.heights()
        ratio = torch.exp(2*(self.log_amplitude-self.log_noise)); noise2 = torch.exp(2*self.log_noise)
        rhs = self.eigen_cross @ heights
        energy = heights @ self.response_gram @ heights
        explained = torch.sum((ratio/(1+ratio*self.values))*rhs.square())
        residual = energy-explained
        if float(residual.detach()) < -1e-7*self.n:
            raise ArithmeticError('GP sufficient-statistic projection produced negative residual energy')
        quadratic = torch.clamp(residual, min=0.)/noise2
        logdet = self.n*torch.log(noise2)+torch.log1p(ratio*self.values).sum()
        jacobian = self.counts @ torch.log(torch.diff(heights)/torch.diff(self.knots))
        smooth_prior = .5*torch.diff(self.log_slopes).square().sum()
        return .5*(logdet+quadratic)-jacobian+smooth_prior

    @torch.no_grad()
    def coefficients(self):
        ratio = torch.exp(2*(self.log_amplitude-self.log_noise))
        return self.axes @ ((ratio/(1+ratio*self.values))*(self.eigen_cross @ self.heights()))


def fit(model, steps=200):
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=.03)
    best_loss, best, history = float('inf'), None, []
    for step in range(steps+1):
        optimizer.zero_grad(set_to_none=True)
        loss = model.nll()/model.n
        if not torch.isfinite(loss):
            raise ArithmeticError('Nonfinite global warped-GP density')
        if float(loss.detach()) < best_loss:
            best_loss = float(loss.detach()); best = {k: p.detach().clone() for k, p in model.named_parameters()}
        if step % 20 == 0 or step == steps:
            history.append(dict(step=step, best_nll_per_row=best_loss,
                amplitude=float(torch.exp(model.log_amplitude.detach())), noise=float(torch.exp(model.log_noise.detach()))))
        if step < steps:
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 10.); optimizer.step()
            with torch.no_grad():
                model.log_slopes.clamp_(-5., 5.)
                model.log_amplitude.clamp_(np.log(.05), np.log(10.))
                model.log_noise.clamp_(np.log(.001), np.log(1.))
    with torch.no_grad():
        for k, p in model.named_parameters():
            p.copy_(best[k])
    return history
