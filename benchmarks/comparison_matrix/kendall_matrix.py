import numpy as np
from scipy.stats import kendalltau
from typing import Dict, List
import pandas as pd
from numpy.typing import NDArray

def compute_kendall_matrix(values: Dict[str, np.ndarray]) -> pd.DataFrame:
    measures: List[str] = list(values.keys())
    n = len(measures)
    matrix: NDArray = np.zeros((n, n))
    
    for i, measure1 in enumerate(measures):
        for j, measure2 in enumerate(measures):
            if i <= j:
                tau, _ = kendalltau(values[measure1], values[measure2])
                matrix[i, j] = matrix[j, i] = tau
    
    return pd.DataFrame(
        data=matrix, 
        index=pd.Index(measures), 
        columns=pd.Index(measures)
    )
