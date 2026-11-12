"""Fast blind proposals followed by the exact frozen BC conditional kernel score."""
import json
import numpy as np
import torch
from scipy.linalg import null_space
from scipy.spatial import cKDTree
from scipy.stats import rankdata

import conditional_ui_witness as ui
import conditional_ux
import matrix_function_profiles as profiles
import run_conditional_kernel_moments as kernel

KEY = 'UX_columns_worst_soft_count_eps010'


class Scorer:
    def __init__(self, bank, original, device='cuda'):
        self.data, self.device = bank.data, device
        self.centers = json.loads((original/'protocol.json').read_text())['Ui_kernel_centers']
        self.z = self.data.u-self.data.u.mean(0)
        self.z /= np.sqrt(np.var(self.data.u, axis=0).mean())
        operators = [np.load(original/f'X_witness_column{j}.npy', allow_pickle=False) for j in range(self.data.dx)]
        _, targets, _, _ = profiles.task_modes(bank.responses[:, :1], operators, tolerance=1e-8)
        self.width = max(t.shape[1] for t in targets)
        values = np.zeros((self.data.n, self.data.dx, self.width))
        for j, target in enumerate(targets):
            values[:, j, :target.shape[1]] = target
        self.flat = torch.tensor(values.reshape(self.data.n, -1), dtype=torch.float64, device=device)
        self.norm = torch.tensor(np.sum(values*values, axis=-1), dtype=torch.float64, device=device)
        with np.load(original/'W_only_geometry.npz', allow_pickle=False) as saved:
            directions = saved['directions']
        with np.load(original/'signed_conditional_moments.npz', allow_pickle=False) as saved:
            energy = saved['Ui_innovation_with_context_raw'][0]
        self.exact_cache = [(v.copy(), e.copy()) for v, e in zip(directions, energy)]
        self.proxy_cache, self.audits = [], []

    def direction(self, v, exact=True, force=False):
        v = np.asarray(v)/np.linalg.norm(v)
        cache = self.exact_cache if exact else self.proxy_cache
        if not force:
            match = next((e for old, e in cache if min(np.linalg.norm(v-old), np.linalg.norm(v+old)) < 1e-6), None)
            if match is not None:
                return match
        if exact:
            features, audit = ui.features(self.z, v, seed=430, centers=self.centers)
            feature = features['Ui_innovation_with_context']
            self.audits.append(dict(direction=v.tolist(), **audit))
        else:
            # Only a proposal mechanism. Final comparisons always use the
            # original nested-regression feature construction above.
            p = (rankdata(self.z @ v)-.5)/len(self.z)
            marginal, _ = ui.nystrom(p[:, None], 16, 430); marginal -= marginal.mean(0)
            rest = self.z @ null_space(v[None])
            context, _ = ui.nystrom(rest, 16, 431); context -= context.mean(0)
            context /= max(float(np.mean(np.sum(context*context, axis=1)))**.5, 1e-30)
            ids = cKDTree(rest).query(rest, k=min(65, len(rest)))[1][:, 1:]
            residual = marginal-marginal[ids].mean(1); residual -= residual.mean(0)
            context = np.column_stack((np.ones(len(rest)), context))
            feature = (residual[:, :, None]*context[:, None]).reshape(len(rest), -1)
            feature /= max(float(np.mean(np.sum(feature*feature, axis=1)))**.5, 1e-30)
        raw, _ = kernel.moments(torch.tensor(feature, dtype=torch.float64, device=self.device), self.flat, self.norm, self.width)
        cache.append((v.copy(), raw.copy()))
        return raw

    def score(self, q, exact=True):
        graph = np.array([self.direction(v, exact=exact) for v in q.T])
        return conditional_ux.sparsity(np.maximum(graph, 0.))[KEY]
