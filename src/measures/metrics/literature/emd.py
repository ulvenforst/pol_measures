import numpy as np
from scipy.stats import wasserstein_distance
from ...base import PolarizationMeasure

class EMDPolSciPy(PolarizationMeasure):
    def _create_target_distribution(self, n: int) -> np.ndarray:
        """Create bimodal distribution with 0.5 mass at extremes."""
        target = np.zeros(n)
        target[0] = target[-1] = 0.5
        return target

    def compute(self, x: np.ndarray, weights: np.ndarray) -> float:
        weights = weights / np.sum(weights)
        target_weights = self._create_target_distribution(len(x))
        emd = wasserstein_distance(x, x, weights, target_weights)
        return 0.5 - emd

class EMDPol(PolarizationMeasure):
    """
    Compute the Earth Mover's Distance for a given distribution of weights and positions,
    constructing an optimal aimed distribution.

    Parameters:
        x (np.ndarray): The positions of the distribution (Is not necessary; added for testing).
        weights (np.ndarray): Weights of the distribution.

    Returns:
        float: The Earth Mover's Distance between a distribution to its consensus.
    """
    
    def compute(self, x: np.ndarray, weights: np.ndarray) -> float:
        n = len(weights)
        weights = weights / np.sum(weights)
        
        left_mass = right_mass = 0.5
        total_cost = 0

        # Move from left
        for i, target_mass in enumerate(weights):
            if left_mass > 0:
                mass_to_move = min(left_mass, target_mass)
                cost = mass_to_move * i / (n - 1)
                total_cost += cost
                left_mass -= mass_to_move
                target_mass -= mass_to_move

            if target_mass > 0 and left_mass == 0:
                break

        # Move from right
        for i in range(n-1, -1, -1):
            target_mass = weights[i]
            if right_mass > 0:
                mass_to_move = min(right_mass, target_mass)
                cost = mass_to_move * (n - 1 - i) / (n - 1)
                total_cost += cost
                right_mass -= mass_to_move
                target_mass -= mass_to_move

            if target_mass > 0 and right_mass == 0:
                break

        return 0.5 - total_cost


if __name__ == "__main__":
   # Crear instancia de la medida
   emd = EMDPol()
   
   # Definir casos de prueba
   print("\nPruebas con escala de 5 puntos:")
   x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
   
   # Caso 1: Distribución uniforme
   w1 = np.ones(5) / 5
   print("\nCaso 1 - Distribución uniforme:")
   print(f"Polarización: {emd(x, w1):.6f}")
   
   # Caso 2: Distribución bimodal perfecta
   w2 = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
   print("\nCaso 2 - Distribución bimodal perfecta:")
   print(f"Polarización: {emd(x, w2):.6f}")  # Debería dar 0.5
   
   # Caso 3: Distribución unimodal en el centro
   w3 = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
   print("\nCaso 3 - Distribución unimodal centrada:")
   print(f"Polarización: {emd(x, w3):.6f}")
   
   # Caso 4: Distribución sesgada a la izquierda
   w4 = np.array([0.5, 0.3, 0.1, 0.1, 0.0])
   print("\nCaso 4 - Distribución sesgada a la izquierda:")
   print(f"Polarización: {emd(x, w4):.6f}")
   
   # Caso 5: Distribución bimodal asimétrica
   w5 = np.array([0.4, 0.1, 0.0, 0.1, 0.4])
   print("\nCaso 5 - Distribución bimodal asimétrica:")
   print(f"Polarización: {emd(x, w5):.6f}")
   
   # Pruebas con diferentes tamaños de escala
   print("\nPruebas con diferentes tamaños de escala:")
   
   # Escala de 3 puntos
   x3 = np.array([0.0, 0.5, 1.0])
   w3_bimodal = np.array([0.5, 0.0, 0.5])
   print("\nEscala de 3 puntos (bimodal):")
   print(f"Polarización: {emd(x3, w3_bimodal):.6f}")
   
   # Escala de 7 puntos
   x7 = np.linspace(0, 1, 7)
   w7_bimodal = np.array([0.3, 0.2, 0.0, 0.0, 0.0, 0.2, 0.3])
   print("\nEscala de 7 puntos (bimodal):")
   print(f"Polarización: {emd(x7, w7_bimodal):.6f}")
