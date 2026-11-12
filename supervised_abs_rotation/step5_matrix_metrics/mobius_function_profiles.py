"""All-order conditional function dividends, without input independence claims."""
import numpy as np


def dividends(values):
    """Fast subset Moebius transform; array index is the input coalition mask."""
    count=len(values);d=count.bit_length()-1
    if count!=2**d:raise ValueError('Expected one value for every coalition')
    out=values.clone() if hasattr(values,'clone') else np.array(values,copy=True)
    for i in range(d):
        for mask in range(count):
            if mask&(1<<i):out[mask]-=out[mask^(1<<i)]
    return out


def allocate_shapley(parts):
    count=len(parts);d=count.bit_length()-1
    out=[]
    for i in range(d):
        out.append(sum(parts[mask]/mask.bit_count() for mask in range(1,count) if mask&(1<<i)))
    if hasattr(parts,'clone'):
        import torch
        return torch.stack(out)
    return np.stack(out)
