"""Finite predictive-skill games, exact Shapley allocation and unassigned risk.

The optional monotone envelope permits each coalition to ignore extra inputs.
It is a finite-estimator definition, not a claim of calibrated Sobol effects.
"""
import math

import numpy as np


def allocate(raw_skill,d):
    raw=np.asarray(raw_skill,dtype=float)
    if raw.shape[0]!=2**d or not np.isfinite(raw).all():
        raise ValueError('Supply one finite predictive skill for every coalition')
    clipped=np.clip(raw,0.,1.);clipped[0]=0.
    game=clipped.copy()
    for i in range(d):
        for mask in range(2**d):
            if mask&(1<<i):game[mask]=np.maximum(game[mask],game[mask^(1<<i)])
    effects=np.zeros((d+1,*raw.shape[1:]))
    for i in range(d):
        for mask in range(2**d):
            if mask&(1<<i):continue
            weight=1/(d*math.comb(d-1,mask.bit_count()))
            effects[i]+=weight*(game[mask|(1<<i)]-game[mask])
    effects[-1]=1-game[-1]
    error=float(np.max(np.abs(effects.sum(0)-1)))
    if error>1e-12 or np.min(effects)<-1e-12:
        raise ArithmeticError('Predictive skill allocation lost nonnegativity or conservation')
    return effects,game,dict(maximum_conservation_error=error,
        mean_monotone_envelope_increase=float(np.mean(game-clipped)),
        maximum_monotone_envelope_increase=float(np.max(game-clipped)),
        fraction_negative_raw_skill=float(np.mean(raw<0)),
        fraction_raw_skill_above_one=float(np.mean(raw>1)))
