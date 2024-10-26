import numpy as np
from ...base import PolarizationMeasure

class BiPolMeasure(PolarizationMeasure):
    def compute(self, x: np.ndarray, weights: np.ndarray) -> float:
        mu = np.average(x, weights=weights)

        if np.all(x[weights > 0] == mu):
            return 0

        L = x < mu
        R = ~L

        mean_diff = (np.average(x[R], weights=weights[R]) - 
                     np.average(x[L], weights=weights[L]))

        return 4 * weights[L].sum() * weights[R].sum() * mean_diff
