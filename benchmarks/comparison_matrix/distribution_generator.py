import numpy as np
import math
from typing import Iterator, Tuple
from itertools import combinations_with_replacement

def generate_distributions(n: int, k: int = 5) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate all possible distributions of n elements in k bins using Stars and Bars method.
    Each distribution is normalized and paired with equidistant x values in [0,1].
    
    Parameters:
        n (int): Number of elements to distribute (population size)
        k (int): Number of bins (default 5 for Likert scale)
    
    Yields:
        Tuple[np.ndarray, np.ndarray]: (x, weights) where x is bin positions and weights 
        are normalized frequencies
    """
    x = np.linspace(0, 1, k)  # Bin positions
    
    # Generate all possible distributions
    for combination in combinations_with_replacement(range(k), n):
        # Count frequencies
        weights = np.zeros(k)
        for i in combination:
            weights[i] += 1
        
        # Normalize weights
        weights = weights / n
        
        yield x, weights

def count_distributions(n: int, k: int = 5) -> int:
    """
    Calculate number of possible distributions using Stars and Bars formula.
    
    Parameters:
        n (int): Number of elements
        k (int): Number of bins
        
    Returns:
        int: Number of possible distributions
    """
    return int(math.comb(n + k - 1, k - 1))
