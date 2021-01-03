"""Algorithms for sampling from the unit simplex."""

import numpy as np
import sys


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


def kraemer_sampling(n, N=1):
    """Sample from an n-dimensional simplex using Kraemer's algorithm."""
    M = sys.maxsize
    P = []

    for _ in range(N):

        rng = np.random.default_rng()

        X = rng.choice(M - 1, replace=False, size=n - 1)
        X = sorted(X)
        X = [0] + X + [M]
        Y = np.diff(X)

        # Store one of the generated samples, which is now guaranteed to
        # lie on the unit simplex by construction.
        P.append(np.asarray(Y) / M)

    return P


# HIC SVNT DRACONES

P = kraemer_sampling(3, 20000)

for x, y, _ in P:
    print(x, y)
