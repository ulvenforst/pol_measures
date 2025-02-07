import numpy as np
from typing import Dict, List
from src.measures.metrics.literature import EMDPol, EstebanRay, Experts, ShannonPol, VanDerEijkPol
from src.measures.metrics.proposed import MEC, BiPol
from src.measures.metrics.proposed.mec import MECNormalized

class MeasureCalculator:
    def __init__(self, tolerance: float = 1e-4):
        """
        Initialize all polarization measures.
        
        Parameters:
            tolerance (float): Numerical tolerance for rounding values
        """
        self.tolerance = tolerance
        self.measures = {
            'MEC(1,1)': MEC(alpha=1, beta=1),
            'MEC(2,1.15)N': MECNormalized(),
            'MEC(2,1.15)': MEC(),
            'MEC(1,2)': MEC(alpha=1, beta=2),
            'MEC(2,2)': MEC(alpha=2, beta=2),
            'ER(1.6)': EstebanRay(alpha=1.6),
            'ER(0.8)': EstebanRay(),
            'EMD': EMDPol(),
            'Experts': Experts(),
            'Shannon': ShannonPol(),
            'VanDerEijk': VanDerEijkPol(),
            'BiPol': BiPol()
        }
        self.results: Dict[str, List[float]] = {name: [] for name in self.measures}
        
    def calculate_all(self, x: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """Calculate all measures for a given distribution."""
        values = {name: measure(x, weights) for name, measure in self.measures.items()}
        return {name: np.round(val/self.tolerance)*self.tolerance 
               for name, val in values.items()}
    
    def process_distribution(self, x: np.ndarray, weights: np.ndarray) -> None:
        results = self.calculate_all(x, weights)
        for name, value in results.items():
            self.results[name].append(value)
    
    def get_values(self) -> Dict[str, np.ndarray]:
        return {name: np.array(values) for name, values in self.results.items()}
