import unittest

import numpy as np

from generator import generate, metrics


class GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.data = np.random.default_rng(1).normal(-3, 1.5, 1000)

    def test_values(self):
        sample = generate(self.data, 500, np.random.default_rng(2))
        self.assertTrue(np.isin(sample, self.data).all())

    def test_seed(self):
        x = generate(self.data, 200, np.random.default_rng(42))
        y = generate(self.data, 200, np.random.default_rng(42))
        np.testing.assert_array_equal(x, y)

    def test_mean(self):
        sample = generate(self.data, 100000, np.random.default_rng(3))
        self.assertLess(abs(metrics(sample)['mean'] - metrics(self.data)['mean']), 0.03)


if __name__ == '__main__':
    unittest.main()
