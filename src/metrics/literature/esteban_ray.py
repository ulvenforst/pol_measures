from ...base import ParametricPolarizationMeasure
from typing import Optional
import numpy as np

class EstebanRayMeasure(ParametricPolarizationMeasure):
    def __init__(self, alpha: float = 1.6, K: Optional[float] = None) -> None:
        if not 0 < alpha <= 1.6:
            raise ValueError("alpha must be in (0, 1.6]")
        
        super().__init__(alpha=alpha, K=K)
    
    def compute(self, x: np.ndarray, weights: np.ndarray) -> float:
        weights = weights / np.sum(weights)
        
        K = self.parameters['K']
        if K is None:
            K = 1 / (2 * ((0.5) ** (2 + self.parameters['alpha'])))
        
        return (K * 
                np.sum(weights ** (1 + self.parameters['alpha']) * 
                      weights[:, None] * 
                      np.abs(x[:, None] - x)))

if __name__ == "__main__":
    # Crear instancia con valores por defecto
    er = EstebanRayMeasure()
    
    # Algunos casos de prueba
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    # Caso 1: Distribución uniforme
    w1 = np.ones(5) / 5
    print("\nCaso 1 - Distribución uniforme:")
    print(f"Polarización: {er(x, w1):.6f}")
    
    # Caso 2: Distribución bimodal
    w2 = np.array([0.4, 0.1, 0.0, 0.1, 0.4])
    print("\nCaso 2 - Distribución bimodal:")
    print(f"Polarización: {er(x, w2):.6f}")
    
    # Caso 3: Distribución sesgada
    w3 = np.array([0.5, 0.3, 0.1, 0.1, 0.0])
    print("\nCaso 3 - Distribución sesgada:")
    print(f"Polarización: {er(x, w3):.6f}")
    
    # Probar con diferentes valores de alpha
    print("\nProbando diferentes valores de alpha en distribución bimodal:")
    for alpha in [0.5, 1.0, 1.6]:
        er_alpha = EstebanRayMeasure(alpha=alpha)
        print(f"alpha = {alpha:.1f}: {er_alpha(x, w2):.6f}")
    
    # Probar con un K específico
    print("\nProbando con K específico (K=1.0):")
    er_custom = EstebanRayMeasure(K=1.0)
    print(f"Polarización: {er_custom(x, w2):.6f}")
