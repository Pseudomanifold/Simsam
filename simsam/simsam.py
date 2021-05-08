"""Algorithms for sampling from the unit simplex."""

import numpy as np
import sys


def naive_sampling(n, N=1):
    """Sample from an n-dimensional simplex using a naive algorithm.

    This is a naive sampling procedure that will not result in
    samples being drawn uniformly from the n-simplex. However,
    it is fast and might be suitable as a baseline algorithm.

    Parameters
    ----------
    n : int
        Dimension of simplex to sample from.
    N : int
        Number of points to sample.

    Returns
    -------
    `np.array` of shape (N, n), with the rows corresponding to
    individual samples.
    """
    P = []

    for _ in range(N):
        a = np.asarray([np.random.uniform() for _ in range(n - 1)])
        p = [a[i] * np.prod(1 - a[:i]) for i in range(n - 1)]
        p.append(1 - np.sum(p))

        # Append sample to list of all samples that are to be returned
        # by this function.
        P.append(p)

    return np.asarray(P, dtype=float)


def kraemer_sampling(n, N=1, full_support=True):
    """Sample from an n-dimensional simplex using Kraemer's algorithm.

    This is the preferred algorithm for sampling uniformly from the unit
    n-simplex. Following Smith & Tromble _[1] the sampling procedure was
    updated to ensure that the sampling is uniform across distributions,
    provided they have full support. This behaviour can be changed using
    the `full_support` parameter.

    Parameters
    ----------
    n : int
        Dimension of simplex to sample from.
    N : int
        Number of points to sample.
    full_support : bool
        If set, only provides distributions with full support, i.e. with
        non-zero values only.

    Returns
    -------
    `np.array` of shape (N, n), with the rows corresponding to
    individual samples.

    .. [1] Noah A. Smith and Roy W. Tromble, "Sampling Uniformly from
    the Unit Simplex," 2004.
    """
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

    return np.asarray(P)


# HIC SVNT DRACONES

P = naive_sampling(3, 20000)

for x, y, _ in P:
    print(x, y)
