from benchmarks.comparison_matrix import (
    generate_distributions, 
    count_distributions,
    MeasureCalculator,
    compute_kendall_matrix,
    plot_correlation_matrix
)

def main(n: int = 10, k: int = 5):
    """
    Run complete comparison analysis.
    
    Parameters:
        n (int): Population size
        k (int): Number of bins (default 5 for Likert scale)
    """
    # Inicializar calculador
    calculator = MeasureCalculator()
    
    # Contar y mostrar número total de distribuciones
    total_distributions = count_distributions(n, k)
    print(f"Analyzing {total_distributions} distributions...")
    
    # Procesar cada distribución
    for i, (x, weights) in enumerate(generate_distributions(n, k)):
        calculator.process_distribution(x, weights)
        if (i + 1) % 1000 == 0:  # Mostrar progreso cada 1000 distribuciones
            print(f"Processed {i + 1}/{total_distributions} distributions")
    
    # Obtener rankings y calcular matriz de correlaciones
    rankings = calculator.get_rankings()
    # print(f"Distribution {calculator.distributions}")   
    # print(f"Results {calculator.results}")
    # print(f"Rankings {rankings}")
    correlation_matrix = compute_kendall_matrix(rankings)
    
    # Visualizar resultados
    print("\nKendall's tau correlation matrix:")
    print(correlation_matrix)
    
    plot_correlation_matrix(
        correlation_matrix,
        title=f"Kendall's tau correlations (n={n}, k={k})"
    )

if __name__ == "__main__":
    # Empezar con una población pequeña para pruebas
    main(n=10)
