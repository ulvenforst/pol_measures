import numpy as np
from ...base import ParametricPolarizationMeasure
from ...validation import validate_parameters

class Comete(ParametricPolarizationMeasure):
    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        super().__init__(alpha=alpha, beta=beta)
        self.precision = 1e-4
        self.tests_per_iter = 10
        
    def _pol_aux(self, x: np.ndarray, weights: np.ndarray, y: float) -> float:
        return np.sum((weights ** self.parameters['alpha']) * 
                     (np.abs(x - y) ** self.parameters['beta']))
    
    def _find_minimum(self, x: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
        low = np.min(x)
        high = np.max(x)
        
        while high - low > self.precision:
            lin = np.linspace(low, high, self.tests_per_iter)
            f_lin = [self._pol_aux(x, weights, y) for y in lin]
            i = int(np.argmin(f_lin))
            low = lin[max(0, i - 2)]
            high = lin[min(len(lin) - 1, i + 2)]
            
        y_star = (low + high) / 2
        return y_star, self._pol_aux(x, weights, y_star)
    
    def compute(self, x: np.ndarray, weights: np.ndarray) -> float:
        validate_parameters(**self.parameters)
        _, polarization = self._find_minimum(x, weights)
        return polarization

if __name__ == "__main__":
   # Crear instancias con diferentes parámetros
   comete_default = Comete()  # alpha=beta=1.0 por defecto
   comete_custom = Comete(alpha=1.2, beta=1.2)
   
   print("\nPruebas con escala de 5 puntos:")
   # x = np.linspace(0, 1, 5)
   x = np.array([1, 2, 3, 4, 5])
   
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
