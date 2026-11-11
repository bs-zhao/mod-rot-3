"""Support of generic joint U/X polynomial projection coefficients."""
from collections import defaultdict

import numpy as np

import kernel_anova as ka
import surrogates as sur
import tail_support as ts


class Structure:
    def __init__(self, powers, d, seed=391):
        self.powers = powers
        self.d = d
        self.degree = int(powers.sum(1).max())
        self.up = sur.powers_for(d, self.degree)
        self.uw = sur.weights_for(self.up)
        lookup = {tuple(p): i for i, p in enumerate(self.up)}
        self.blocks = defaultdict(list)
        for i, p in enumerate(powers):
            self.blocks[int(p[:d].sum()), tuple(p[d:])].append((lookup[tuple(p[:d])], i))
        self.blocks = {key: sorted(value) for key, value in self.blocks.items()}
        self.probes = np.random.default_rng(seed).normal(size=(4*len(self.up), d))
        self.has_u = powers[:, :d].sum(1) > 0
        self.has_x = powers[:, d:].sum(1) > 0
        self.cross_u = (powers[:, :d] > 0).sum(1) >= 2

    def rotate(self, coef, q):
        transform = ka.rotation_representation(self.up, self.uw, q, self.probes)
        out = np.empty_like(coef)
        for pairs in self.blocks.values():
            uids, rows = map(list, zip(*pairs))
            out[rows] = transform[np.ix_(uids, uids)] @ coef[rows]
        return out

    def scores(self, coef):
        energy = np.asarray(coef, dtype=float)**2
        out = {}
        groups = dict(all_U=energy[self.has_u], UX_terms=energy[self.has_u & self.has_x],
                      UU_cross_terms=energy[self.cross_u])
        u_groups, ux_groups, x_columns = defaultdict(float), defaultdict(float), defaultdict(list)
        for p, value, has_u, has_x in zip(self.powers, energy, self.has_u, self.has_x):
            if not has_u:
                continue
            u = tuple(np.flatnonzero(p[:self.d]))
            u_groups[u] += value
            if has_x:
                x = tuple(np.flatnonzero(p[self.d:]))
                ux_groups[u, x] += value
                x_columns[tuple(p[self.d:])].append(value)
        groups["U_support_groups"] = np.asarray(list(u_groups.values()))
        groups["UX_support_groups"] = np.asarray(list(ux_groups.values()))
        for group, values in groups.items():
            for metric, loss in ts.scores(values).items():
                out[group+"_"+metric] = loss
        columns = [ts.scores(v) for v in x_columns.values() if sum(v) > 0]
        if columns:
            for key in columns[0]:
                out["UX_columns_"+key] = float(np.mean([v[key] for v in columns]))
        out["control_coefficient_U_fro2"] = float(energy[self.has_u].sum())
        return out
