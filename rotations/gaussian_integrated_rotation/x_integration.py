"""X probability measure: exact immutable 500-bank or declared generator.

For CR3/CR5, their special-condition fixed bank is a DISCRETE measure;
we never fit/smooth a continuous X density or silently sample CR4's
uniform distribution. For CR4, continuous generator reads the ACTUAL S0
design manifest. Future independent uniform generators can use a strict
user-supplied JSON distribution spec; compound designs require extending
the sampler, not pretending all six coordinates are independent.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from .data import sha256


def _read_spec(source,explicit=None):
    if explicit:
        path=Path(explicit).expanduser().resolve()
        if not path.is_file():raise FileNotFoundError(path)
        spec=json.loads(path.read_text(encoding="utf-8"))
        if spec.get("family")!="independent_uniform":
            raise ValueError("Only explicitly declared independent_uniform generator is supported")
        low=np.asarray(spec["low"],np.float64)
        high=np.asarray(spec["high"],np.float64)
        desc=f"explicit:{path}"
        fingerprint=sha256(path)
    elif source["model"]=="CR4":
        path=source["bank"].parent/"CR4_3P_design.json"
        if not path.exists():
            raise FileNotFoundError("CR4 generative-X needs the actual S0 design JSON")
        spec=json.loads(path.read_text(encoding="utf-8"))
        if spec.get("bank_sha256")!=source["hashes"]["bank"] or spec.get("independent_X") is not True:
            raise RuntimeError("CR4 generator metadata does not verify independent X bank")
        alo,ahi=map(float,spec["amount_range"])
        plo,phi=map(float,spec["probability_range"])
        low=np.array([alo,alo,plo,alo,alo,plo])
        high=np.array([ahi,ahi,phi,ahi,ahi,phi])
        desc="CR4_actual_S0_six_independent_uniform_generator"
        fingerprint=sha256(path)
    else:
        raise ValueError(f"{source['model']} has a special FIXED X bank; "
                         "do not assume independent continuous X. "
                         "Use x_mode=bank, or supply an explicit --x-spec-file.")
    if low.shape!=(6,) or high.shape!=(6,) or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("Independent-uniform X requires six finite bounds")
    if not np.all(high>low) or low[[2,5]].min()<0 or high[[2,5]].max()>1:
        raise ValueError("X range invalid; probabilities must lie in [0,1]")
    return low,high,desc,fingerprint


def x_nodes(source,mode="bank",n=512,seed=1,bank_limit=500,
            x_spec_file=None):
    if mode=="bank":
        x=source["bank_x"]
        group=np.asarray(source["group_codes"],int)
        if not 1<=int(bank_limit)<=len(x):
            raise ValueError("bank_limit must be in [1,500]")
        if bank_limit<len(x):
            chosen=np.sort(np.random.default_rng(seed).choice(
                len(x),int(bank_limit),replace=False))
            x=x[chosen]
            group=group[chosen]
        else:
            chosen=np.arange(len(x))
        weights=np.full(len(x),1./len(x),dtype=float)
        meta=dict(mode="exact_discrete_bank_average",count=len(x),
                  original_bank_size=len(source["bank_x"]),
                  selected_bank_ids=chosen.tolist(),
                  bank_sha256=source["hashes"]["bank"],
                  categories=source["group_names"],seed=seed)
        return x,weights,group,meta
    if mode=="generative":
        if n<2 or n&(n-1):
            raise ValueError("Generative X Sobol count must be a power of 2 >=2")
        low,high,desc,fingerprint=_read_spec(source,x_spec_file)
        u=qmc.Sobol(d=6,scramble=True,seed=seed).random_base2(int(np.log2(n)))
        x=low+(high-low)*u
        if not np.isfinite(x).all():
            raise RuntimeError("Bad Sobol X nodes")
        weights=np.full(n,1./n,dtype=float)
        groups=np.zeros(n,dtype=int)
        meta=dict(mode="generative_sobol",distribution=desc,
                  spec_sha256=fingerprint,count=n,seed=seed,
                  low=low.tolist(),high=high.tolist(),categories=["all"])
        return x,weights,groups,meta
    raise ValueError("X mode must be bank or generative")
