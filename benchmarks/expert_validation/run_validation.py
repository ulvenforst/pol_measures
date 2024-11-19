import numpy as np
from typing import Dict
import pandas as pd
from src.measures.metrics.literature import EMDPol, EstebanRay, Experts, ShannonPol, VanDerEijkPol
from src.measures.metrics.proposed import MEC, BiPol
from scipy.stats import kendalltau
from .data import ValidationData
import matplotlib.pyplot as plt

class ValidationCalculator:
    def __init__(self):
        self.measures = {
            'MEC(1,1)': MEC(alpha=1, beta=1),
            'MEC(1.2,1.2)': MEC(),
            'MEC(2,1.2)': MEC(alpha=2),
            'MEC(1.2,2)': MEC(beta=2),
            'MEC(2,2)': MEC(alpha=2, beta=2),
            'EMD': EMDPol(),
            'ER(1.6)': EstebanRay(),
            'ER(0.8)': EstebanRay(alpha=0.8),
            'Experts': Experts(),
            'Shannon': ShannonPol(),
            'VanDerEijk': VanDerEijkPol(),
            'BiPol': BiPol(),
        }
        self.results: Dict[str, list] = {name: [] for name in self.measures}

    def process_distributions(self, x_values: np.ndarray, distributions: np.ndarray) -> None:
        """Procesa todas las distribuciones en orden"""
        for dist in distributions:
            for name, measure in self.measures.items():
                self.results[name].append(measure(x_values, dist))

    def get_values(self) -> Dict[str, np.ndarray]:
        return {name: np.array(values) for name, values in self.results.items()}

def get_ranking_from_order(order: list) -> np.ndarray:
    """
    Convierte un orden en ranking.
    El ranking[i-1] indica en qué posición aparece i en el orden.
    """
    ranking = np.zeros(len(order))
    for rank, dist_idx in enumerate(order, 1):
        ranking[dist_idx-1] = rank
    return ranking

def save_rankings_and_values(measure_values: Dict[str, np.ndarray], filename: str = "rankings_y_valores.txt"):
    """
    Guarda los valores y rankings de cada medida en un archivo.
    """
    # Orden original según expertos
    original_order = [3,10,1,12,13,15,5,6,9,11,2,7,8,4,14]
    original_ranking = get_ranking_from_order(original_order)
    
    with open(filename, 'w') as f:
        f.write("RANKINGS Y VALORES DE POLARIZACIÓN\n")
        f.write("="*50 + "\n\n")
        
        # Primero el orden original
        f.write("Orden original según expertos:\n")
        f.write(f"Orden: {original_order}\n")
        f.write(f"Ranking: {list(original_ranking)}\n\n")
        
        # Para cada medida
        for measure_name, values in measure_values.items():
            f.write(f"\n{measure_name}:\n")
            f.write("-"*30 + "\n")
            
            # Valores de polarización en orden original
            f.write("Valores de polarización (orden original):\n")
            for i, val in enumerate(values):
                f.write(f"Distribución {i+1}: {val:.6f}\n")
            
            # Obtener el orden que produce esta medida
            value_order = np.argsort(values) + 1
            
            # Obtener el ranking
            ranking = get_ranking_from_order(value_order)
                
            f.write("\nOrden producido:\n")
            f.write(f"{list(value_order)}\n")
            f.write("\nRanking resultante:\n")
            f.write(f"{list(ranking)}\n")
            
            # Calcular y mostrar correlación
            tau, _ = kendalltau(original_ranking, ranking)
            f.write(f"\nCorrelación de Kendall con orden original: {tau:.6f}\n")
            f.write("\n" + "="*50 + "\n")

def main():
    # Cargar datos
    data = ValidationData()
    distributions = data.get_normalized_distributions()
    x_values = data.x_values
    
    # Calcular todas las medidas
    calculator = ValidationCalculator()
    calculator.process_distributions(x_values, distributions)
    measure_values = calculator.get_values()
    
    # Añadir los valores exactos de los expertos
    real_mean_scores = np.array([
        16.9655,  # Distribución 1
        35.8966,  # Distribución 2
        6.6207,   # Distribución 3
        70.0517,  # Distribución 4
        26.8621,  # Distribución 5
        27.0517,  # Distribución 6
        49.5862,  # Distribución 7
        59.0517,  # Distribución 8
        28.7586,  # Distribución 9
        13.1207,  # Distribución 10
        30.5000,  # Distribución 11
        21.8103,  # Distribución 12
        22.7069,  # Distribución 13
        85.9483,  # Distribución 14
        22.8966   # Distribución 15
    ])
    measure_values['RealMeanExp'] = real_mean_scores
    
    # Guardar rankings y valores detallados
    save_rankings_and_values(measure_values)
    
    # Calcular y visualizar correlaciones
    original_order = [3,10,1,12,13,15,5,6,9,11,2,7,8,4,14]
    original_ranking = get_ranking_from_order(original_order)
    
    correlations = {}
    for measure_name, values in measure_values.items():
        value_order = np.argsort(values) + 1
        ranking = get_ranking_from_order(value_order)
        tau, _ = kendalltau(original_ranking, ranking)
        correlations[measure_name] = tau
    
    correlations = pd.Series(correlations).sort_values(ascending=False)
    print("\nCorrelaciones con orden original:")
    print(correlations)
    
    # Visualizar
    plt.figure(figsize=(12, 6))
    ax = correlations.plot(kind='bar')
    plt.title('Correlación de cada medida con el orden de los expertos')
    plt.xlabel('Medidas')
    plt.ylabel('Correlación de Kendall tau')
    
    # Añadir valores sobre las barras
    for i, v in enumerate(correlations):
        ax.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
