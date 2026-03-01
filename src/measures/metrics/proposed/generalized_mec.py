from typing import Callable, Optional, cast

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.optimize._optimize import OptimizeResult

from ...base import ParametricPolarizationMeasure
from ...validation import validate_parameters


class GeneralizedMEC(ParametricPolarizationMeasure):
    """
    Generalized MEC measure with configurable alienation function.

    GMEC_{alpha,f}(M) = min_y sum_i w_i^alpha * f(|x_i - y|)
    """

    def __init__(
        self,
        alpha: float = 2.0,
        alienation: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        super().__init__(alpha=alpha)
        self._alienation = alienation if alienation is not None else (lambda d: d)

    def _apply_alienation(self, distances: np.ndarray) -> np.ndarray:
        values = np.asarray(self._alienation(distances), dtype=np.float64)
        if values.shape != distances.shape:
            try:
                values = np.broadcast_to(values, distances.shape)
            except ValueError as exc:
                raise ValueError(
                    "Alienation function output shape is not compatible."
                ) from exc

        if not np.all(np.isfinite(values)):
            raise ValueError("Alienation function must return finite values.")
        if np.any(values < 0):
            raise ValueError("Alienation function must be non-negative.")

        return values

    def compute(self, x: np.ndarray, weights: np.ndarray) -> float:
        alpha = self.parameters["alpha"]
        validate_parameters(alpha=alpha)

        weights_alpha = weights**alpha

        def obj_func(y: float) -> float:
            transformed = self._apply_alienation(np.abs(x - y))
            return float(np.sum(weights_alpha * transformed))

        result = cast(
            OptimizeResult, minimize_scalar(obj_func, bounds=(0, 1), method="bounded")
        )

        return float(result.fun)
