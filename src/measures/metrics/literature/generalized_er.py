from typing import Callable, Optional

import numpy as np

from ...base import ParametricPolarizationMeasure


class GeneralizedER(ParametricPolarizationMeasure):
    """
    Generalized Esteban-Ray measure with configurable alienation function.

    ER_{alpha,f}(w,x) = K * sum_i sum_j w_i^{1+alpha} * w_j * f(|x_i - x_j|)

    The standard ER corresponds to f(d) = d. This class allows swapping in
    different alienation functions (power, polynomial, exponential, etc.)
    as proposed in "Revisiting the Measurement of Polarization".

    The normalization constant K is computed so that the extreme bimodal
    distribution (50/50 split at the endpoints) yields a value of 1:
        K = 1 / (2 * 0.5^{2+alpha} * f(d_max))
    where d_max = x_max - x_min (the diameter of the support).
    """

    def __init__(
        self,
        alpha: float = 0.8,
        alienation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        K: Optional[float] = None,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        self._alienation = alienation if alienation is not None else lambda d: d
        self._K = K
        super().__init__(alpha=alpha)

    def compute(self, x: np.ndarray, weights: np.ndarray) -> float:
        alpha = self.parameters["alpha"]
        distances = np.abs(x[:, None] - x)
        f_distances = self._alienation(distances)

        K = self._K
        if K is None:
            d_max = float(np.max(x) - np.min(x))
            f_dmax = float(self._alienation(np.array(d_max)))
            K = 1.0 / (2.0 * (0.5 ** (2.0 + alpha)) * f_dmax)

        W_i = (weights ** (1 + alpha))[:, None]
        W_j = weights[None, :]

        return float(K * np.sum(W_i * W_j * f_distances))
