import numpy as np


def make_rng_streams(seed: int, names):
    names = tuple(names)
    children = np.random.SeedSequence(int(seed)).spawn(len(names))
    return {name: np.random.default_rng(child) for name, child in zip(names, children, strict=True)}
