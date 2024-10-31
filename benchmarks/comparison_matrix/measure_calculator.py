import numpy as np
from typing import Dict, List
from src.measures.metrics.literature import EMDPol, EstebanRay, Experts, ShannonPol, VanDerEijkPol
from src.measures.metrics.proposed import Comete, BiPol

class MeasureCalculator:
    def __init__(self, tolerance: float = 1e-4):
        """
        Initialize all polarization measures.
        
        Parameters:
            tolerance (float): Numerical tolerance for rounding values
        """
        self.tolerance = tolerance
        self.measures = {
            'Comete(1,1)': Comete(alpha=1, beta=1),
            'Comete(1.2,1.2)': Comete(),
            'Comete(2,1)': Comete(alpha=2),
            'Comete(1,2)': Comete(beta=2),
            'Comete(2,2)': Comete(alpha=2, beta=2),
            'EMD': EMDPol(),
            'ER(1.6)': EstebanRay(),
            'ER(0.8)': EstebanRay(alpha=0.8),
            'Experts': Experts(),
            'Shannon': ShannonPol(),
            'VanDerEijk': VanDerEijkPol(),
            'BiPol': BiPol()
        }
        self.results: Dict[str, List[float]] = {name: [] for name in self.measures}
        
    def calculate_all(self, x: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        values = {name: measure(x, weights) for name, measure in self.measures.items()}
        return {name: np.round(val/self.tolerance)*self.tolerance 
               for name, val in values.items()}
    
    def process_distribution(self, x: np.ndarray, weights: np.ndarray) -> None:
        results = self.calculate_all(x, weights)
        for name, value in results.items():
            self.results[name].append(value)
    
    def get_values(self) -> Dict[str, np.ndarray]:
        return {name: np.array(values) for name, values in self.results.items()}
