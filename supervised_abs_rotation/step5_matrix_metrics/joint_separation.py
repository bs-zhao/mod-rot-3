"""Mixed U/X tensor-block separation of a frozen joint projection.

Reference: isotropic Gaussian U times empirical univariate X marginals.
This is a product-reference projection descriptor, not a claim of observed
input independence or a Sobol decomposition of the dependent data.
"""
import math

import numpy as np

import tail_support as ts


def marginal_basis(gram):
    """Degree-ordered empirical orthogonal polynomials, allowing finite support."""
    degree = len(gram)-1
    coefficients = []
    for p in range(degree+1):
        vector = np.eye(degree+1)[p]
        # Reorthogonalize to handle finite/categorical marginal support.
        for _ in range(2):
            for q in coefficients:
                vector -= q * (q @ gram @ vector)
        norm = float(vector @ gram @ vector)
        if norm > 1e-10*max(float(gram[p, p]), np.finfo(float).tiny):
            coefficients.append(vector/np.sqrt(norm))
        else:
            coefficients.append(np.zeros(degree+1))
    out = gram @ np.asarray(coefficients).T
    # Degree ordering implies exact triangularity; eliminate roundoff leakage.
    out[np.triu_indices(degree+1, 1)] = 0.
    return out


def gaussian_moments(degree, sigma):
    out = np.zeros((degree+1, degree+1))
    for p in range(degree+1):
        for k in range(p+1):
            if (p-k) % 2 == 0:
                r = (p-k)//2
                out[p, k] = math.factorial(p)*sigma**p/(2**r*math.factorial(r)*math.sqrt(math.factorial(k)))
    return out


class JointSeparation:
    def __init__(self, powers, weights, d, gram_train, n_train):
        self.powers, self.weights, self.d = powers, weights, d
        self.degree = int(powers.sum(1).max())
        lookup = {tuple(p): i for i, p in enumerate(powers)}
        self.x_moments, self.x_ranks = [], []
        for j in range(d, powers.shape[1]):
            indices = []
            for degree in range(self.degree+1):
                p = np.zeros(powers.shape[1], dtype=int); p[j] = degree
                indices.append(lookup[tuple(p)])
            g = gram_train[np.ix_(indices, indices)]/n_train
            g = g/weights[indices, None]/weights[None, indices]
            one = marginal_basis(g)
            self.x_moments.append(one)
            self.x_ranks.append(int(np.count_nonzero(np.diag(one))))
        self.partitions = []
        for i in range(d):
            for j in range(d, powers.shape[1]):
                left = powers[:, [i, j]]; right = np.delete(powers, [i, j], axis=1)
                lp, li = np.unique(left, axis=0, return_inverse=True)
                rp, ri = np.unique(right, axis=0, return_inverse=True)
                self.partitions.append((i, j-d, li, ri, len(lp), len(rp),
                                        int(np.flatnonzero(lp.sum(1) == 0)[0]),
                                        int(np.flatnonzero(rp.sum(1) == 0)[0])))

    def moment_map(self, sigma):
        moments = [gaussian_moments(self.degree, sigma)]*self.d + self.x_moments
        transform = np.ones((len(self.powers), len(self.powers)))
        for j, one in enumerate(moments):
            transform *= one[self.powers[:, j, None], self.powers[None, :, j]]
        return transform

    def scores(self, orthogonal_coef):
        groups = {'UX_block_full': [], 'UX_block_centered': []}
        audit = []
        for i, j, li, ri, nl, nr, l0, r0 in self.partitions:
            matrix = np.zeros((nl, nr)); matrix[li, ri] = orthogonal_coef
            full = np.linalg.svd(matrix, compute_uv=False)**2
            matrix[l0, :] = 0.; matrix[:, r0] = 0.
            centered = np.linalg.svd(matrix, compute_uv=False)**2
            for group, spectrum in zip(groups, (full, centered)):
                # Only remove machine-level singular values relative to block energy.
                spectrum[spectrum < spectrum.max(initial=0.)*1e-14] = 0.
                groups[group].append(ts.scores(spectrum))
            audit.append(dict(u_axis=i, x_axis=j, full_spectrum=full.tolist(),
                              centered_spectrum=centered.tolist()))
        out = {}
        for group, table in groups.items():
            for metric in table[0]:
                values = [row[metric] for row in table]
                out[group+'_mean_'+metric] = float(np.mean(values))
                out[group+'_worst_'+metric] = float(np.max(values))
        out['control_product_reference_norm2'] = float(orthogonal_coef @ orthogonal_coef)
        return out, audit
