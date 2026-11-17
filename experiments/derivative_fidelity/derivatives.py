"""Uniform field engine for ALL order 1-3 derivative blocks.

Indices always appear in the order printed in BLOCKS. Dense field storage is
memory-mapped to disk. No change to the frozen network or data.
"""
from __future__ import annotations
import math
from pathlib import Path
from itertools import product
from functools import lru_cache
import numpy as np
import torch
from torch.func import jacfwd, jacrev, vmap

BLOCKS = ("U", "X", "UU", "UX", "XX", "UUU", "UUX", "UXX", "XXX")
BLOCKS_BY_ORDER = {k: len(k) for k in BLOCKS}


def block_shape(block, d, p):
    return tuple(d if c=="U" else p for c in block)


def derivative_function(h, block, mode="mixed"):
    if block not in BLOCKS:
        raise ValueError(block)
    if mode not in ("mixed", "forward", "reverse"):
        raise ValueError(mode)
    # derivative output axes follow block letter order
    fn = jacrev(h, 0 if block[0]=="U" else 1) if mode in ("mixed", "reverse") else jacfwd(h, 0 if block[0]=="U" else 1)
    for letter in block[1:]:
        op = jacrev if mode=="reverse" else jacfwd
        fn = op(fn, 0 if letter=="U" else 1)
    return fn


def flatten_grid(V, X):
    """Return vectors of common evaluation coordinates for a (S,K) grid."""
    S = len(V)
    if X.ndim == 2:
        K = len(X)
        return np.repeat(V, K, axis=0), np.tile(X, (S, 1)), S, K
    if X.ndim == 3 and X.shape[0] == S:
        K = X.shape[1]
        return np.repeat(V, K, axis=0), X.reshape(-1, X.shape[-1]), S, K
    raise ValueError("X shape must be (K,p) or (S,K,p)")


def eval_batched(fn, V, X, device="cpu", dtype=torch.float64, chunk=256):
    v, x, S, K = flatten_grid(V, X)
    pieces = []
    for a in range(0, len(v), chunk):
        vv = torch.as_tensor(v[a:a+chunk], device=device, dtype=dtype)
        xx = torch.as_tensor(x[a:a+chunk], device=device, dtype=dtype)
        with torch.enable_grad():
            y = vmap(fn)(vv, xx)
        pieces.append(y.detach().cpu().numpy())
    return np.concatenate(pieces).reshape(S, K, *pieces[0].shape[1:])


def write_field(path, h, V, X, block, method="mixed", precision="float64",
                device="cpu", chunk=128, pointwise_fn=None, subject_mask=None):
    """Write a dense field to .npy without retaining all points in RAM."""
    dtype = torch.float64 if precision=="float64" else torch.float32
    v, x, S, K = flatten_grid(V, X)
    shape = (S,K,*block_shape(block, V.shape[1], X.shape[-1]))
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(dest, mode="w+", dtype=np.float64, shape=shape)
    fn = pointwise_fn if pointwise_fn is not None else derivative_function(h, block, method)
    view = out.reshape(len(v), *shape[2:])
    for i in range(0, len(v), chunk):
        a = torch.as_tensor(v[i:i+chunk], dtype=dtype, device=device)
        b = torch.as_tensor(x[i:i+chunk], dtype=dtype, device=device)
        with torch.enable_grad():
            ans = vmap(fn)(a, b)
        view[i:i+len(a)] = ans.detach().cpu().numpy()
    out.flush()
    return str(dest)


def tensor_symmetry_error(field, block):
    a = np.asarray(field)
    if len(block) < 2:
        return 0.0
    # Only axes with the same variable type must agree after swapping.
    axes = [(i,j) for i in range(len(block)) for j in range(i+1,len(block)) if block[i]==block[j]]
    if not axes:
        return 0.0
    err = []
    for i,j in axes:
        perm = list(range(a.ndim)); perm[2+i],perm[2+j]=perm[2+j],perm[2+i]
        err.append(np.max(np.abs(a-np.transpose(a,perm))))
    return float(max(err))


def chain_rule_sigmoid(eta, gradient, hessian, third=None):
    """Return probability derivatives; useful for independent channel self-tests."""
    e = np.asarray(eta)
    p = 1/(1+np.exp(-np.clip(e,-700,700)))
    s1=p*(1-p); s2=s1*(1-2*p); s3=s1*(1-6*p+6*p*p)
    g=s1[...,None]*gradient
    H=s1[...,None,None]*hessian+s2[...,None,None]*np.einsum("...i,...j->...ij",gradient,gradient)
    if third is None:
        return g,H
    t = (s1[...,None,None,None]*third +
         s2[...,None,None,None]*(np.einsum("...ij,...k->...ijk",hessian,gradient)+
                                  np.einsum("...ik,...j->...ijk",hessian,gradient)+
                                  np.einsum("...jk,...i->...ijk",hessian,gradient))+
         s3[...,None,None,None]*np.einsum("...i,...j,...k->...ijk",gradient,gradient,gradient))
    return g,H,t
