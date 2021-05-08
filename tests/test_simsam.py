import numpy as np

from simsam import naive_sampling
from simsam import kraemer_sampling

from unittest import TestCase


class SmokeTest(TestCase):
    """Simple smoke test to ensure that samples are valid."""

    def test(self):
        for f in [naive_sampling, kraemer_sampling]:
            P = f(10, 1000)

            for p in P:
                self.assertAlmostEqual(np.sum(p), 1.0)
