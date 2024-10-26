from ...base import PolarizationMeasure
import numpy as np

class ExpertsMeasure(PolarizationMeasure):
    """
    Expert-based polarization measure for 5-category Likert scales.
    
    Based on a study where 60 experts rated polarization in 15 Likert-like
    distributions, yielding the formula:
    P(n) = (2.14*n₂n₄ + 2.70(n₁n₄ + n₂n₅) + 3.96*n₁n₅)/(0.0099*n²)
    where nᵢ is the frequency of category i.
    """
    
    def compute(self, x: np.ndarray, weights: np.ndarray) -> float:
        if len(x) != 5:
            raise ValueError("Experts measure was designed only for 5-category histograms")
            
        n2n4_term = 2.14 * weights[1] * weights[3]
        n1n4_n2n5_term = 2.70 * (weights[0] * weights[3] + weights[1] * weights[4])
        n1n5_term = 3.96 * weights[0] * weights[4]
        
        numerator = n2n4_term + n1n4_n2n5_term + n1n5_term
        denominator = 0.0099 * (np.sum(weights) ** 2)
        
        return (numerator / denominator) / 100

if __name__ == "__main__":
    # Crear instancia de la medida
    expert = ExpertsMeasure()
    
    # Definir algunos casos de prueba
    x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    # Caso 1: Distribución con tendencia a la izquierda
    weights1 = np.array([2, 1, 1, 1, 0])
    print("\nCaso 1 - Distribución con tendencia a la izquierda:")
    print(f"Polarización: {expert(x, weights1):.6f}")
    
    # Caso 2: Distribución con tendencia a la derecha
    weights2 = np.array([0, 1, 1, 0, 3])
    print("\nCaso 2 - Distribución con tendencia a la derecha:")
    print(f"Polarización: {expert(x, weights2):.6f}")
    
    # Caso 3: Distribución uniforme
    weights3 = np.ones(5)
    print("\nCaso 3 - Distribución uniforme:")
    print(f"Polarización: {expert(x, weights3):.6f}")
    
    # Caso 4: Distribución bimodal
    weights4 = np.array([2, 0, 1, 0, 2])
    print("\nCaso 4 - Distribución bimodal:")
    print(f"Polarización: {expert(x, weights4):.6f}")
