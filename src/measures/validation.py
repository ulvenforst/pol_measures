from typing import Tuple
import numpy as np

def minmax_normalize_x(x: np.ndarray) -> np.ndarray:
    """Normalize x values to [0,1] range."""
    x_min, x_max = np.min(x), np.max(x)
    if x_min == x_max:
        return np.zeros_like(x)
    return (x - x_min) / (x_max - x_min)

def validate_histogram(x: np.ndarray, 
                      weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    if x.shape != weights.shape:
        raise ValueError("x and weights must have the same shape")
        
    if x.size < 2:
        raise ValueError("At least two points are required")
        
    if not np.all(np.diff(x) > 0):
        raise ValueError("x values must be strictly increasing")
        
    if np.any(weights < 0):
        raise ValueError("All weights must be non-negative")
        
    if not np.any(weights > 0):
        raise ValueError("At least one weight must be positive")
    
    weights = weights / np.sum(weights)
    x = minmax_normalize_x(x)
    
    return x, weights

def validate_parameters(**parameters) -> None:
    """Validate measure-specific parameters."""
    for name, value in parameters.items():
        if not isinstance(value, (int, float)):
            raise TypeError(f"Parameter {name} must be numeric")
        if value <= 0:
            raise ValueError(f"Parameter {name} must be positive")
