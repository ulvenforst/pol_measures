from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from .validation import validate_histogram

class PolarizationMeasure(ABC):
    """Base class for all polarization measures."""

    def __init__(self) -> None:
        self._cached_result: Optional[float] = None

    @abstractmethod
    def compute(self, x: np.ndarray, weights: np.ndarray) -> float:
        """Compute the polarization measure."""
        pass

    def __call__(self, x: np.ndarray, weights: np.ndarray) -> float:
        x, weights = validate_histogram(x, weights)
        self._cached_result = self.compute(x, weights)
        return self._cached_result

    @property
    def last_result(self) -> Optional[float]:
        return self._cached_result

class ParametricPolarizationMeasure(PolarizationMeasure):
    """Base class for polarization measures with parameters."""

    def __init__(self, **parameters) -> None:
        super().__init__()
        self.parameters = parameters

    def update_parameters(self, **parameters) -> None:
        self.parameters.update(parameters)
        self._cached_result = None
