# `Simsam`: Simplex Sampling Methods

This small package implements methods for sampling from a unit simplex,
a problem that often crops up in a data analysis context.

## Usage

There is only a single sampling strategy that results in uniform samples
from the unit simplex:

```python
from simsam import kraemer_sampling

# Sample 1,000 points from the 10-dimensional unit simplex.
dim = 10
N = 1000
samples = kraemer_sampling(dim, N)
```

For comparison purposes, there is also a naive sampling procedure, which
does *not* result in uniform samples.

```python
from simsam import naive_sampling

# Sample 1,000 points from the 10-dimensional unit simplex. Notice that
# the samples will be biased.
dim = 10
N = 1000
samples = naive_sampling(dim, N)
```
