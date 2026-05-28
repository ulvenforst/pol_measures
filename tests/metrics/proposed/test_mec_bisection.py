import unittest

import numpy as np

from src.measures.metrics.proposed.mec import MEC
from src.measures.utils.optimization import mec_bisection_value


class TestMECBisectionValue(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

    def assertMatchesScipyMEC(
        self,
        x: np.ndarray,
        weights: np.ndarray,
        *,
        alpha: float = 2.0,
        beta: float = 1.15,
        rtol: float = 1e-4,
        atol: float = 1e-8,
    ):
        expected = MEC(alpha=alpha, beta=beta).compute(x, weights)
        actual = mec_bisection_value(
            x,
            weights,
            alpha=alpha,
            beta=beta,
            epsilon=1e-10,
        )
        self.assertTrue(
            np.isclose(actual, expected, rtol=rtol, atol=atol),
            msg=(
                f"bisection MEC {actual} did not match SciPy MEC {expected} "
                f"for alpha={alpha}, beta={beta}, x={x}, weights={weights}"
            ),
        )

    def test_matches_existing_mec_standard_cases(self):
        """Reuse the distributions covered by the existing MEC correctness tests."""
        distributions = [
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            np.array([0.5, 0.0, 0.0, 0.0, 0.5]),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
            np.array([0.1, 0.2, 0.4, 0.2, 0.1]),
            np.array([0.4, 0.3, 0.2, 0.1, 0.0]),
            np.array([0.4, 0.1, 0.0, 0.1, 0.4]),
        ]

        for weights in distributions:
            with self.subTest(weights=weights):
                self.assertMatchesScipyMEC(self.x, weights)

    def test_matches_existing_mec_for_multiple_parameters(self):
        weights = np.array([0.4, 0.1, 0.0, 0.1, 0.4])
        parameter_sets = [
            (1.0, 1.001),
            (1.0, 1.15),
            (2.0, 1.15),
            (2.0, 2.0),
            (0.7, 1.4),
            (1.5, 2.5),
        ]

        for alpha, beta in parameter_sets:
            with self.subTest(alpha=alpha, beta=beta):
                self.assertMatchesScipyMEC(
                    self.x,
                    weights,
                    alpha=alpha,
                    beta=beta,
                    rtol=2e-4,
                    atol=1e-8,
                )

    def test_matches_existing_mec_on_nonuniform_support(self):
        x = np.array([0.0, 0.1, 0.35, 0.9, 1.0])
        weights = np.array([0.05, 0.25, 0.4, 0.2, 0.1])
        self.assertMatchesScipyMEC(x, weights, alpha=2.0, beta=1.15)

    def test_matches_existing_mec_on_deterministic_random_sweep(self):
        rng = np.random.default_rng(20260528)

        for support_size in [3, 5, 10, 20, 50]:
            for _ in range(20):
                interior = np.sort(rng.random(max(support_size - 2, 0)))
                x = np.concatenate(([0.0], interior, [1.0]))
                weights = rng.random(support_size)
                weights /= weights.sum()

                with self.subTest(support_size=support_size):
                    self.assertMatchesScipyMEC(
                        x,
                        weights,
                        alpha=2.0,
                        beta=1.15,
                        rtol=5e-4,
                        atol=1e-8,
                    )

    def test_rejects_invalid_parameters_and_histograms(self):
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        with self.assertRaises(ValueError):
            mec_bisection_value(self.x, weights, alpha=0.0, beta=1.15)
        with self.assertRaises(ValueError):
            mec_bisection_value(self.x, weights, alpha=2.0, beta=1.0)
        with self.assertRaises(ValueError):
            mec_bisection_value(self.x, weights, alpha=2.0, beta=1.15, epsilon=0.0)
        with self.assertRaises(ValueError):
            mec_bisection_value(self.x[::-1], weights, alpha=2.0, beta=1.15)
        with self.assertRaises(ValueError):
            mec_bisection_value(self.x, np.zeros_like(weights), alpha=2.0, beta=1.15)
        with self.assertRaises(ValueError):
            mec_bisection_value(self.x, -weights, alpha=2.0, beta=1.15)


if __name__ == "__main__":
    unittest.main()
