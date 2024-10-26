import numpy as np
from typing import Dict, List, Tuple
from src.measures.metrics.literature import EMDPol, EstebanRay, Experts, ShannonPol, VanDerEijkPol
from src.measures.metrics.proposed import Comete, BiPol

class MeasureCalculator:
    """Class to calculate and store polarization values for different measures."""
    
    def __init__(self):
        """Initialize all polarization measures."""
        self.measures = {
            'Comete(1,1)': Comete(),
            'Comete(1.2,1.2)': Comete(alpha=1.2, beta=1.2),
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
        # Para almacenar resultados
        self.results: Dict[str, List[float]] = {name: [] for name in self.measures}
        self.distributions: List[Tuple[np.ndarray, np.ndarray]] = []
        
    def calculate_all(self, x: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """
        Calculate polarization using all measures for a single distribution.
        
        Parameters:
            x (np.ndarray): Bin positions
            weights (np.ndarray): Distribution weights
                
        Returns:
            Dict[str, float]: Dictionary with measure names and their values
        """
        return {name: measure(x, weights) for name, measure in self.measures.items()}
    
    def process_distribution(self, x: np.ndarray, weights: np.ndarray) -> None:
        """
        Process a single distribution and store results.
        
        Parameters:
            x (np.ndarray): Bin positions
            weights (np.ndarray): Distribution weights
        """
        results = self.calculate_all(x, weights)
        self.distributions.append((x, weights))
        for name, value in results.items():
            self.results[name].append(value)
    
    def get_rankings(self, tolerance: float = 1e-4) -> Dict[str, np.ndarray]:
        """
        Get rankings for all measures, considering values within tolerance as equal.
        
        Parameters:
            tolerance (float): Numerical tolerance for considering two values equal
        
        Returns:
            Dict[str, np.ndarray]: Dictionary with measure names and arrays of distribution indices
            sorted by their polarization values (from least to most polarized)
        """
        rankings = {}
        for name, values in self.results.items():
            # Redondear valores según tolerancia
            rounded_values = np.round(np.array(values)/tolerance)*tolerance
            # Obtener índices que ordenarían los valores
            rankings[name] = np.argsort(rounded_values)
        
        return rankings
    
    def clear(self) -> None:
        """Clear stored results."""
        self.results = {name: [] for name in self.measures}
        self.distributions = []
