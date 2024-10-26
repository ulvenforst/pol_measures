from ...base import PolarizationMeasure
import numpy as np
from scipy.stats import entropy

class ShannonPol(PolarizationMeasure):
   """
   Polarization measure based on Shannon entropy using a consensus-based approach.
   
   Computes polarization as 1 - (1 + sum(pᵢ * log₂(1 - |xᵢ - μ|/d))),
   where μ is the mean and d is the range of the distribution.
   """
   
   def compute(self, x: np.ndarray, weights: np.ndarray) -> float:
       weights = weights / np.sum(weights)
       mu_x = np.sum(weights * x)
       dx = np.max(x) - np.min(x)
       
       # Añadimos epsilon para evitar log(0)
       consensus = 1 + np.sum(weights * 
                            np.log2(1 - np.abs(x - mu_x) / dx + 
                                   np.finfo(float).eps))
       return 1 - consensus

if __name__ == "__main__":
   # Crear instancias de ambas medidas
   shannon_pol = ShannonPol()
   
   # Definir casos de prueba
   x = np.linspace(0, 1, 5)  # 5 puntos equidistantes
   
   # Caso 1: Distribución uniforme
   w1 = np.ones(5) / 5
   print("\nCaso 1 - Distribución uniforme:")
   print(f"ShannonPol: {shannon_pol(x, w1):.6f}")
   
   # Caso 2: Distribución bimodal perfecta
   w2 = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
   print("\nCaso 2 - Distribución bimodal perfecta:")
   print(f"ShannonPol: {shannon_pol(x, w2):.6f}")
   
   # Caso 3: Distribución unimodal
   w3 = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
   print("\nCaso 3 - Distribución unimodal:")
   print(f"ShannonPol: {shannon_pol(x, w3):.6f}")
   
   # Caso 4: Distribución concentrada en un punto
   w4 = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
   print("\nCaso 4 - Distribución concentrada:")
   print(f"ShannonPol: {shannon_pol(x, w4):.6f}")
   
   # Caso 5: Distribución asimétrica
   w5 = np.array([0.4, 0.3, 0.2, 0.1, 0.0])
   print("\nCaso 5 - Distribución asimétrica:")
   print(f"ShannonPol: {shannon_pol(x, w5):.6f}")
