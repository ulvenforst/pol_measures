import numpy as np
from ...base import ParametricPolarizationMeasure
from ...validation import validate_parameters

class Comete(ParametricPolarizationMeasure):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__(alpha=alpha, beta=beta)
        self.precision = 1e-4
        self.tests_per_iter = 10
        
    def _pol_aux(self, x: np.ndarray, weights: np.ndarray, y: float) -> float:
        return np.sum((weights ** self.parameters['alpha']) * 
                     (np.abs(x - y) ** self.parameters['beta']))
    
    def _find_minimum(self, x: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
        low = np.min(x)
        high = np.max(x)
        
        while high - low > self.precision:
            lin = np.linspace(low, high, self.tests_per_iter)
            f_lin = [self._pol_aux(x, weights, y) for y in lin]
            i = int(np.argmin(f_lin))
            low = lin[max(0, i - 2)]
            high = lin[min(len(lin) - 1, i + 2)]
            
        y_star = (low + high) / 2
        return y_star, self._pol_aux(x, weights, y_star)
    
    def compute(self, x: np.ndarray, weights: np.ndarray) -> float:
        validate_parameters(**self.parameters)
        _, polarization = self._find_minimum(x, weights)
        return polarization
