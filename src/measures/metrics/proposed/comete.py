import numpy as np
from scipy.optimize import minimize_scalar
from typing import cast
from scipy.optimize._optimize import OptimizeResult

from ...base import ParametricPolarizationMeasure
from ...validation import validate_parameters

class Comete(ParametricPolarizationMeasure):
    """
    Defined as the minimum effort of carrying out a distribution M towards 
    a single point of consensus p.
    """
    
    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__(alpha=alpha, beta=beta)
        self._alpha = alpha
        self._beta = beta
    
    def compute(self, x: np.ndarray, weights: np.ndarray) -> float:
        """
        Compute polarization using scipy's optimization.
        
        Parameters:
            x (np.ndarray): The positions of the distribution
            weights (np.ndarray): The weights of the distribution
            
        Returns:
            float: Polarization value
        """
        validate_parameters(**self.parameters)
        
        weights_alpha = weights ** self._alpha
        
        def obj_func(y: float) -> float:
            return float(np.sum(weights_alpha * (np.abs(x - y) ** self._beta)))
        
        result = cast(OptimizeResult, minimize_scalar(
            obj_func,
            bounds=(0, 1),
            method='bounded'
        ))
        
        return float(result.fun)

if __name__ == "__main__":
   # Crear instancias con diferentes parámetros
   comete_default = Comete()  # alpha=beta=1.0 por defecto
   comete_custom = Comete(alpha=1.2, beta=1.2)
   
   print("\nPruebas con escala de 5 puntos:")
   x = np.linspace(0, 1, 5)
   # x = np.array([1, 2, 3, 4, 5])
   
   # Caso 1: Distribución uniforme
   w1 = np.ones(5) / 5
   print("\nCaso 1 - Distribución uniforme:")
   print(f"Polarización (default params): {comete_default(x, w1):.6f}")
   print(f"Polarización (alpha=beta=1.2): {comete_custom(x, w1):.6f}")
   
   # Caso 2: Distribución bimodal perfecta
   w2 = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
   print("\nCaso 2 - Distribución bimodal perfecta:")
   print(f"Polarización (default params): {comete_default(x, w2):.6f}")
   print(f"Polarización (alpha=beta=1.2): {comete_custom(x, w2):.6f}")
   
   # Caso 3: Distribución unimodal en el centro
   w3 = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
   print("\nCaso 3 - Distribución unimodal centrada:")
   print(f"Polarización (default params): {comete_default(x, w3):.6f}")
   print(f"Polarización (alpha=beta=1.2): {comete_custom(x, w3):.6f}")
   
   # Caso 4: Distribución sesgada
   w4 = np.array([0.5, 0.3, 0.1, 0.1, 0.0])
   print("\nCaso 4 - Distribución sesgada:")
   print(f"Polarización (default params): {comete_default(x, w4):.6f}")
   print(f"Polarización (alpha=beta=1.2): {comete_custom(x, w4):.6f}")
   
   # Caso 5: Distribución bimodal asimétrica
   w5 = np.array([0.4, 0.1, 0.0, 0.1, 0.4])
   print("\nCaso 5 - Distribución bimodal asimétrica:")
   print(f"Polarización (default params): {comete_default(x, w5):.6f}")
   print(f"Polarización (alpha=beta=1.2): {comete_custom(x, w5):.6f}")
   
   # Pruebas con diferentes tamaños de escala
   print("\nPruebas con diferentes tamaños de escala:")
   
   # Escala de 3 puntos
   x3 = np.array([0.0, 0.5, 1.0])
   w3_bimodal = np.array([0.5, 0.0, 0.5])
   print("\nEscala de 3 puntos (bimodal):")
   print(f"Polarización (default params): {comete_default(x3, w3_bimodal):.6f}")
   print(f"Polarización (alpha=beta=1.2): {comete_custom(x3, w3_bimodal):.6f}")
   
   # Escala de 7 puntos
   x7 = np.linspace(0, 1, 7)
   w7_bimodal = np.array([0.3, 0.2, 0.0, 0.0, 0.0, 0.2, 0.3])
   print("\nEscala de 7 puntos (bimodal):")
   print(f"Polarización (default params): {comete_default(x7, w7_bimodal):.6f}")
   print(f"Polarización (alpha=beta=1.2): {comete_custom(x7, w7_bimodal):.6f}")
