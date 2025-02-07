from benchmarks.comparison_matrix import (
   generate_distributions, 
   count_distributions,
   MeasureCalculator,
   compute_kendall_matrix,
   plot_correlation_matrix
)

def main(n: int = 5, k: int = 5):
    calculator = MeasureCalculator()
    total_distributions = count_distributions(n, k)
    print(f"Analyzing {total_distributions} distributions...")

    for i, (x, weights) in enumerate(generate_distributions(n, k)):
        calculator.process_distribution(x, weights)
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{total_distributions} distributions")

    values = calculator.get_values()
    correlation_matrix = compute_kendall_matrix(values)
    
    print("\nKendall's tau correlation matrix:")
    print(correlation_matrix)
    plot_correlation_matrix(correlation_matrix, title=f"Kendall's tau correlations (n={n}, k={k})")

if __name__ == "__main__":
    main(n=100)
