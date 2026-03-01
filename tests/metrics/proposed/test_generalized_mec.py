import unittest

import numpy as np

from src.measures.metrics.proposed.generalized_mec import GeneralizedMEC
from src.measures.metrics.proposed.mec import MEC


class TestGeneralizedMEC(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    def test_alpha_validation(self):
        with self.assertRaises(ValueError):
            GeneralizedMEC(alpha=0.0)

    def test_equivalence_with_mec_power_alienation(self):
        """GeneralizedMEC with f(d)=d^beta should match MEC(alpha,beta)."""
        alpha = 2.0
        beta = 1.15
        weights = np.array([0.4, 0.1, 0.0, 0.1, 0.4])

        gmec = GeneralizedMEC(alpha=alpha, alienation=lambda d: d**beta)
        mec = MEC(alpha=alpha, beta=beta)

        self.assertAlmostEqual(gmec(self.x, weights), mec(self.x, weights), places=8)

    def test_single_point_distribution(self):
        weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        measure = GeneralizedMEC(alpha=2.0, alienation=lambda d: d**2)
        self.assertAlmostEqual(measure(self.x, weights), 0.0, places=8)

    def test_rejects_negative_alienation_values(self):
        measure = GeneralizedMEC(alpha=1.0, alienation=lambda d: d - 1.0)
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        with self.assertRaises(ValueError):
            measure(self.x, weights)


if __name__ == "__main__":
    unittest.main()
