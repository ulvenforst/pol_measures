import numpy as np
import math
from typing import Iterator, Tuple
from itertools import combinations_with_replacement

def generate_distributions(n: int, k: int = 5) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Genera todas las distribuciones posibles de n elementos en k contenedores (bins)
    usando el método Stars and Bars, pero omitiendo las distribuciones espejo.
    
    Cada distribución se normaliza y se empareja con posiciones equidistantes en [0,1].
    Se genera únicamente la forma canónica, es decir, aquella para la cual
    tuple(w) <= tuple(w[::-1]).
    
    Parameters:
        n (int): Número de elementos a distribuir (tamaño de la masa)
        k (int): Número de contenedores (por defecto 5 para una escala de Likert)
    
    Yields:
        Tuple[np.ndarray, np.ndarray]: (x, weights) donde x son las posiciones y 
        weights la distribución normalizada de frecuencias.
    """
    x = np.linspace(0, 1, k)
    
    for combination in combinations_with_replacement(range(k), n):
        weights = np.zeros(k)
        for i in combination:
            weights[i] += 1
        weights = weights / n
        
        if tuple(weights) <= tuple(weights[::-1]):
            yield x, weights

def count_distributions(n: int, k: int = 5) -> int:
    """
    Calcula la cantidad de distribuciones posibles usando la fórmula Stars and Bars.
    
    Parameters:
        n (int): Número de elementos
        k (int): Número de contenedores
        
    Returns:
        int: Cantidad de distribuciones posibles
    """
    return int(math.comb(n + k - 1, k - 1))

