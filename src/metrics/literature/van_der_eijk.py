from ...base import PolarizationMeasure
import numpy as np

class VanDerEijkMeasure(PolarizationMeasure):
    """
    Van Der Eijk's agreement measure adapted as a polarization measure.
    
    The measure decomposes empirical distributions into ideal-type distributions
    and calculates agreement based on patterns of unimodality and multimodality.
    Finally converts agreement to polarization.
    """
    
    def _pattern_vector(self, V: np.ndarray) -> np.ndarray:
        """Create a pattern vector marking positive frequencies as 1."""
        P = np.zeros_like(V)
        P[V > 0] = 1
        return P
    
    def _minnz(self, V: np.ndarray) -> float:
        """Calculate the smallest non-zero value in the vector."""
        non_zero_values = V[V > 0]
        if non_zero_values.size == 0:
            print("Warning: Minimum calculation failed. No non-zero elements found.")
            return np.inf
        return np.min(non_zero_values)
    
    def _pattern_agreement(self, P: np.ndarray) -> float:
        """Calculate the agreement score from a pattern vector."""
        K = len(P)
        TDU = TU = 0

        for i in range(K-2):
            for j in range(i+1, K-1):
                for m in range(j+1, K):
                    if P[i] == 1 and P[j] == 0 and P[m] == 1:
                        TDU += 1  # 101 pattern, bimodal
                    if P[i] == 1 and P[j] == 1 and P[m] == 0:
                        TU += 1   # 110 pattern, unimodal
                    if P[i] == 0 and P[j] == 1 and P[m] == 1:
                        TU += 1   # 011 pattern, unimodal

        if TU == TDU == 0:
            U = 1
        else:
            U = ((K-2) * TU - (K-1) * TDU) / ((K-2) * (TU + TDU))

        S = np.sum(P)
        A = U * (1 - (S - 1) / (K - 1))
        
        if np.isnan(A):
            A = 0
        if S == 1:
            A = 1

        return A
    
    def compute(self, x: np.ndarray, weights: np.ndarray) -> float:
        if len(weights) < 3:
            print("Warning: length of vector < 3, measure is not defined.")
            return float('nan')
            
        if np.min(weights) < 0:
            raise ValueError("Error: negative values found in frequency vector.")

        AA = 0
        N = np.sum(weights)
        R = np.array(weights, dtype=float)

        for _ in range(len(weights)):
            P = self._pattern_vector(R)
            if np.max(P) == 0:
                break
                
            A = self._pattern_agreement(P)
            m = self._minnz(R)
            L = P * m
            w = np.sum(L) / N
            AA += w * A
            R -= L
            
        return 1 - (1 + AA) * 0.5

if __name__ == "__main__":
    # Crear instancia de la medida
    veijk = VanDerEijkMeasure()
    
    # Casos de prueba
    print("\nPruebas con escala de 5 puntos:")
    x = np.linspace(0, 1, 5)
    
    # Caso 1: Distribución uniforme
    w1 = np.ones(5)
    print("\nCaso 1 - Distribución uniforme:")
    print(f"Polarización: {veijk(x, w1):.6f}")
    
    # Caso 2: Distribución bimodal perfecta
    w2 = np.array([5, 0, 0, 0, 5])
    print("\nCaso 2 - Distribución bimodal perfecta:")
    print(f"Polarización: {veijk(x, w2):.6f}")
    
    # Caso 3: Distribución unimodal
    w3 = np.array([1, 2, 4, 2, 1])
    print("\nCaso 3 - Distribución unimodal:")
    print(f"Polarización: {veijk(x, w3):.6f}")
    
    # Caso 4: Distribución con un solo punto
    w4 = np.array([0, 0, 10, 0, 0])
    print("\nCaso 4 - Distribución concentrada:")
    print(f"Polarización: {veijk(x, w4):.6f}")
    
    # Caso 5: Distribución bimodal asimétrica
    w5 = np.array([4, 1, 0, 1, 4])
    print("\nCaso 5 - Distribución bimodal asimétrica:")
    print(f"Polarización: {veijk(x, w5):.6f}")
    
    # Caso 6: Ejemplo original del paper
    w6 = np.array([10, 15, 10, 20, 25, 20])
    x6 = np.linspace(0, 1, 6)
    print("\nCaso 6 - Ejemplo del paper:")
    print(f"Polarización: {veijk(x6, w6):.6f}")

