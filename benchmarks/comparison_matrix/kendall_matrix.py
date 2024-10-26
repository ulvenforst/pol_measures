import numpy as np
from scipy.stats import kendalltau
from typing import Dict, List
import pandas as pd
from numpy.typing import NDArray

def compute_kendall_matrix(rankings: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Compute matrix of Kendall's tau correlations between all measures.
    
    Parameters:
        rankings (Dict[str, np.ndarray]): Dictionary of rankings for each measure
        
    Returns:
        pd.DataFrame: Matrix of Kendall's tau correlations
    """
    measures: List[str] = list(rankings.keys())
    n = len(measures)
    matrix: NDArray = np.zeros((n, n))
    
    for i, measure1 in enumerate(measures):
        for j, measure2 in enumerate(measures):
            if i <= j:
                tau, _ = kendalltau(rankings[measure1], rankings[measure2])
                matrix[i, j] = matrix[j, i] = tau
    
    return pd.DataFrame(
        data=matrix, 
        index=pd.Index(measures), 
        columns=pd.Index(measures)
    )
