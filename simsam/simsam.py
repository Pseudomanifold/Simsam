"""Algorithms for sampling from the unit simplex."""

import numpy as np


def naive_sampling(n, N=1):
    """Sample from an n-dimensional simplex using a naive algorithm."""
    P = []

    for _ in range(N):
        a = np.asarray([np.random.uniform() for _ in range(n - 1)])
        p = [a[i] * np.prod(1 - a[:i]) for i in range(n - 1)]
        p.append(1 - np.sum(p))

        # Append sample to list of all samples that are to be returned
        # by this function.
        P.append(p)

    return P


# HIC SVNT DRACONES

P = naive_sampling(3, 20000)

for x, y, _ in P:
    print(x, y)
