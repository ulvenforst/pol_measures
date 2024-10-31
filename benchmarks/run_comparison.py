import multiprocessing as mp
from itertools import islice
from functools import partial
from typing import List, Tuple, Dict
import numpy as np
from benchmarks.comparison_matrix import (
   generate_distributions, 
   count_distributions,
   MeasureCalculator,
   compute_kendall_matrix,
   plot_correlation_matrix
)

# def process_batch(batch: List[Tuple[np.ndarray, np.ndarray]], 
#                 measures: Dict[str, object],
#                 tolerance: float) -> Dict[str, List[float]]:
#    """Procesa un lote de distribuciones."""
#    results = {name: [] for name in measures}
#    for x, weights in batch:
#        values = {name: measure(x, weights) for name, measure in measures.items()}
#        rounded = {name: np.round(val/tolerance)*tolerance 
#                  for name, val in values.items()}
#        for name, value in rounded.items():
#            results[name].append(value)
#    return results

# def main(n: int = 5, k: int = 5, n_processes: int = None):
#    if n_processes is None:
#        n_processes = mp.cpu_count()
   
#    # Inicializar calculador para obtener las medidas
#    calculator = MeasureCalculator()
#    measures = calculator.measures
   
#    total_distributions = count_distributions(n, k)
#    print(f"Analyzing {total_distributions} distributions using {n_processes} processes...")
   
#    all_distributions = list(generate_distributions(n, k))
   
#    batch_size = len(all_distributions) // (n_processes * 10) 
#    batch_size = max(1, batch_size)
#    batches = [all_distributions[i:i + batch_size] 
#              for i in range(0, len(all_distributions), batch_size)]
   
#    with mp.Pool(n_processes) as pool:
#        process_fn = partial(process_batch, 
#                           measures=measures, 
#                           tolerance=calculator.tolerance)
       
#        results_list = []
#        total_batches = len(batches)
#        for i, batch_result in enumerate(pool.imap(process_fn, batches)):
#            results_list.append(batch_result)
#            if (i + 1) % max(1, total_batches // 100) == 0: 
#                print(f"Processed {i+1}/{total_batches} batches "
#                      f"({((i+1)/total_batches*100):.1f}%)")
   
#    combined_results = {name: [] for name in measures}
#    for batch_result in results_list:
#        for name in measures:
#            combined_results[name].extend(batch_result[name])
   
#    final_values = {name: np.array(values) 
#                   for name, values in combined_results.items()}
   
#    correlation_matrix = compute_kendall_matrix(final_values)
#    print("\nKendall's tau correlation matrix:")
#    print(correlation_matrix)
#    plot_correlation_matrix(correlation_matrix, 
#                          title=f"Kendall's tau correlations (n={n}, k={k})")


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

# if __name__ == "__main__":
#    main(n=100, n_processes=12)
   
