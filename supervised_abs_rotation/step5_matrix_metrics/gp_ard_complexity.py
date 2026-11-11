"""Nyström GP evidence and axis relevance for normalized function ensembles."""
import numpy as np
import torch


def target_modes(latents, gram, tolerance=1e-6):
    values, vectors = np.linalg.eigh((gram+gram.T)/2)
    order = np.argsort(values)[::-1]; values, vectors = np.maximum(values[order], 0.), vectors[:, order]
    total = float(values.sum())
    if total <= 0:
        raise ValueError('An X contrast has no identifiable U-coupling energy')
    rank = min(int(np.searchsorted(np.cumsum(values)/total, 1-tolerance))+1, len(values))
    target = latents @ (vectors[:, :rank]*np.sqrt(values[:rank]/total))
    return target, dict(rank=rank, retained_fraction=float(values[:rank].sum()/total),
                        relative_squared_reconstruction_error=float(values[rank:].sum()/total))


class Model(torch.nn.Module):
    def __init__(self, u, targets, center_ids, starts=3):
        super().__init__()
        self.u, self.targets = u, targets
        self.center_ids = torch.as_tensor(center_ids, dtype=torch.long, device=u.device)
        groups, tasks = len(u), len(targets)
        self.groups, self.tasks, self.starts = groups, tasks, starts
        self.group_index = torch.arange(groups, device=u.device).repeat(tasks*starts)
        self.task_index = torch.arange(tasks, device=u.device).repeat_interleave(groups).repeat(starts)
        self.start_index = torch.arange(starts, device=u.device).repeat_interleave(tasks*groups)
        initial = torch.tensor([.3, 1., 3.], dtype=torch.float64, device=u.device)
        if starts > len(initial):
            raise ValueError('This prospective initialization policy supports 1-3 starts')
        self.eta = torch.nn.Parameter(initial[self.start_index, None].expand(-1, u.shape[-1]).clone())
        self.log_amplitude = torch.nn.Parameter(torch.zeros(len(self.eta), dtype=torch.float64, device=u.device))
        self.log_noise = torch.nn.Parameter(torch.full((len(self.eta),), np.log(.1), dtype=torch.float64, device=u.device))

    def features(self, ids, rows):
        group = self.group_index[ids]
        x = self.u[group][:, rows]*self.eta[ids, None]
        centers = self.u[group][:, self.center_ids]*self.eta[ids, None]
        def kernel(a, b):
            distance = torch.clamp(a.square().sum(-1)[:, :, None]+b.square().sum(-1)[:, None]-2*a @ b.transpose(1, 2), min=0.)
            return torch.exp(-.5*distance)
        covariance = kernel(centers, centers)
        eye = torch.eye(len(self.center_ids), dtype=torch.float64, device=x.device)
        chol = torch.linalg.cholesky((covariance+covariance.transpose(1, 2))/2+1e-6*eye)
        return torch.linalg.solve_triangular(chol, kernel(x, centers).transpose(1, 2), upper=False).transpose(1, 2)

    def system(self, ids, rows):
        phi = self.features(ids, rows)
        y = self.targets[self.task_index[ids]][:, rows]
        gram = phi.transpose(1, 2) @ phi
        cross = phi.transpose(1, 2) @ y
        noise2 = torch.exp(2*self.log_noise[ids]); amplitude2 = torch.exp(2*self.log_amplitude[ids])
        ratio = amplitude2/noise2
        matrix = torch.eye(gram.shape[-1], dtype=torch.float64, device=gram.device)+ratio[:, None, None]*gram
        chol = torch.linalg.cholesky((matrix+matrix.transpose(1, 2))/2)
        solved = torch.cholesky_solve(cross, chol)
        return phi, y, noise2, ratio, chol, cross, solved

    def nll(self, ids, rows):
        _, y, noise2, ratio, chol, cross, solved = self.system(ids, rows)
        logdet = len(rows)*torch.log(noise2)+2*torch.log(torch.diagonal(chol, dim1=1, dim2=2)).sum(1)
        quadratic = (y.square().sum((1, 2))-ratio*(cross*solved).sum((1, 2)))/noise2
        return .5*(logdet+quadratic)  # common normalizing constant omitted

    @torch.no_grad()
    def prediction_error(self, ids, training, query):
        _, _, _, ratio, _, _, solved = self.system(ids, training)
        phi = self.features(ids, query)
        prediction = phi @ (ratio[:, None, None]*solved)
        target = self.targets[self.task_index[ids]][:, query]
        return (prediction-target).square().sum((1, 2))/torch.clamp(target.square().sum((1, 2)), min=1e-30)

    def snapshot(self):
        return {name: value.detach().clone() for name, value in self.named_parameters()}

    def restore(self, state):
        with torch.no_grad():
            for name, value in self.named_parameters():
                value.copy_(state[name])


def optimize(model, rows, steps, chunk=8, callback=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    best = torch.full((len(model.eta),), torch.inf, dtype=torch.float64, device=model.u.device)
    state = model.snapshot(); history = []
    for step in range(steps+1):
        optimizer.zero_grad(set_to_none=True); losses = torch.empty_like(best)
        for a in range(0, len(best), chunk):
            ids = torch.arange(a, min(a+chunk, len(best)), device=model.u.device)
            loss = model.nll(ids, rows)
            if not torch.isfinite(loss).all():
                raise ArithmeticError('Nonfinite GP evidence; preserve run rather than score it as a sparsity failure')
            losses[ids] = loss.detach()
            if step < steps:
                loss.sum().backward()
        improve = losses < best; best[improve] = losses[improve]
        current = model.snapshot()
        for key in state:
            state[key][improve] = current[key][improve]
        if step % 10 == 0 or step == steps:
            row = dict(step=step, mean_nll_per_subject=float(losses.mean()/len(rows)),
                best_mean_nll_per_subject=float(best.mean()/len(rows)), maximum_eta=float(model.eta.detach().max()))
            history.append(row)
            if callback:
                callback(row, history)
        if step < steps:
            squared = sum(p.grad.reshape(len(best), -1).square().sum(1) for p in model.parameters())
            multiplier = torch.clamp(10./torch.sqrt(squared+1e-30), max=1.)
            for p in model.parameters():
                p.grad.mul_(multiplier.reshape(len(best), *([1]*(p.ndim-1))))
            optimizer.step()
            with torch.no_grad():
                model.eta.clamp_(0., 10.)
                model.log_amplitude.clamp_(np.log(.05), np.log(10.))
                model.log_noise.clamp_(np.log(.001), np.log(1.))
    model.restore(state)
    return best.detach().cpu().numpy(), history
