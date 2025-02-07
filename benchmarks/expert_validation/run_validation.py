import numpy as np
from typing import Dict
import pandas as pd
from src.measures.metrics.literature import EMDPol, EstebanRay, Experts, ShannonPol, VanDerEijkPol
from src.measures.metrics.proposed import MEC, BiPol
from src.measures.metrics.proposed.mec import MECNormalized
from scipy.stats import kendalltau
from .data import ValidationData
import matplotlib.pyplot as plt

class ValidationCalculator:
    def __init__(self):
        self.measures = {
            'MEC(1,1)': MEC(alpha=1, beta=1),
            'MEC(2,1.15)': MECNormalized(),
            'MEC(2,1.2)': MEC(beta=1.2),
            'MEC(2,2)': MEC(alpha=2, beta=2),
            'EMD': EMDPol(),
            'ER(0.8)': EstebanRay(),
            'ER(1.6)': EstebanRay(alpha=1.6),
            'Experts': Experts(),
            'Shannon': ShannonPol(),
            'VanDerEijk': VanDerEijkPol(),
            'BiPol': BiPol(),
        }
        self.results: Dict[str, list] = {name: [] for name in self.measures}

    def process_distributions(self, x_values: np.ndarray, distributions: np.ndarray) -> None:
        """Procesa todas las distribuciones en orden"""
        for i, dist in enumerate(distributions):
            for name, measure in self.measures.items():
                # Truncar a 4 decimales
                value = measure(x_values, dist)
                self.results[name].append(np.trunc(value * 10000) / 10000)

    def get_values(self) -> Dict[str, np.ndarray]:
        return {name: np.array(values) for name, values in self.results.items()}

def save_measure_values(measure_values: Dict[str, np.ndarray], expert_scores: np.ndarray, 
                       distributions: np.ndarray, filename: str = "valores_polarizacion.txt"):
    """Guarda los valores de polarización y sus correlaciones con expert_scores"""
    with open(filename, 'w') as f:
        f.write("VALORES DE POLARIZACIÓN\n")
        f.write("="*50 + "\n\n")
        
        # Mostrar distribuciones originales para verificación
        f.write("Distribuciones originales:\n")
        for i, dist in enumerate(distributions, 1):
            f.write(f"Distribución {i}: {dist}\n")
        f.write("\n" + "="*50 + "\n\n")
        
        f.write("Valores de expertos:\n")
        for i, val in enumerate(expert_scores, 1):
            f.write(f"Distribución {i}: {val:.4f}\n")
        f.write("\n" + "="*50 + "\n")
        
        for measure_name, values in measure_values.items():
            f.write(f"\n{measure_name}:\n")
            f.write("-"*30 + "\n")
            
            f.write("Valores de polarización:\n")
            for i, val in enumerate(values, 1):
                f.write(f"Distribución {i}: {val:.4f}\n")
            
            tau, _ = kendalltau(expert_scores, values)
            f.write(f"\nCorrelación de Kendall con expert_scores: {tau:.4f}\n")
            f.write("\n" + "="*50 + "\n")

def main():
    # Cargar datos
    data = ValidationData()
    distributions = data.get_normalized_distributions()
    x_values = data.x_values
    expert_scores = np.trunc(data.expert_scores * 10000) / 10000  # Truncar a 4 decimales
    
    # Calcular valores de polarización
    calculator = ValidationCalculator()
    calculator.process_distributions(x_values, distributions)
    measure_values = calculator.get_values()
    measure_values['RealMeanExp'] = expert_scores
    
    # Guardar valores detallados
    save_measure_values(measure_values, expert_scores, distributions)
    
    # Calcular correlaciones
    correlations = {}
    for measure_name, values in measure_values.items():
        tau, _ = kendalltau(expert_scores, values)
        correlations[measure_name] = np.trunc(tau * 10000) / 10000
    
    correlations = pd.Series(correlations).sort_values(ascending=False)
    print("\nCorrelaciones con expert_scores:")
    print(correlations)
    
    # Visualizar
    plt.figure(figsize=(12, 6))
    ax = correlations.plot(kind='bar')
    plt.title('Correlation with the opinion of the 60 experts')
    plt.xlabel('Measure')
    plt.ylabel('Kendall rank correlation coefficient')
    
    for i, v in enumerate(correlations):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
