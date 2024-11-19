# import numpy as np
# import time
# from typing import Callable, List, Tuple
# import matplotlib.pyplot as plt
# from src.measures.metrics.proposed import MEC, CometeOptimized

# def generate_test_case(size: int) -> Tuple[np.ndarray, np.ndarray]:
#     """Generate test case of specified size."""
#     x = np.linspace(0, 1, size)
#     weights = np.random.random(size)
#     weights /= np.sum(weights)
#     return x, weights

# def benchmark_function(func: Callable, x: np.ndarray, weights: np.ndarray, 
#                       n_runs: int = 100) -> float:
#     """Benchmark a function with given inputs."""
#     start_time = time.perf_counter()
#     for _ in range(n_runs):
#         func(x, weights)
#     end_time = time.perf_counter()
#     return (end_time - start_time) / n_runs

# def run_benchmarks(sizes: List[int], alphas: List[float], betas: List[float],
#                   n_runs: int = 100) -> dict:
#     """
#     Run benchmarks for both implementations with different input sizes 
#     and parameter combinations.
#     """
#     results = {}
    
#     for alpha, beta in zip(alphas, betas):
#         print(f"\nTesting with α={alpha}, β={beta}")
#         comete_opt = CometeOptimized(alpha=alpha, beta=beta)
#         comete_orig = MEC(alpha=alpha, beta=beta)
        
#         opt_times = []
#         orig_times = []
        
#         for size in sizes:
#             print(f"Testing size {size}...")
#             x, weights = generate_test_case(size)
            
#             opt_time = benchmark_function(comete_opt, x, weights, n_runs)
#             orig_time = benchmark_function(comete_orig, x, weights, n_runs)
            
#             opt_times.append(opt_time)
#             orig_times.append(orig_time)
            
#             print(f"Optimized implementation: {opt_time:.6f} seconds")
#             print(f"Original implementation: {orig_time:.6f} seconds")
#             print("-" * 50)
            
#         results[f"α={alpha},β={beta}"] = {
#             'optimized': opt_times,
#             'original': orig_times
#         }
    
#     return results

# def plot_results(sizes: List[int], results: dict) -> None:
#     """Plot benchmark results."""
#     n_params = len(results)
#     fig, axs = plt.subplots(n_params, 1, figsize=(12, 6*n_params))
#     if n_params == 1:
#         axs = [axs]
    
#     for (param_set, times), ax in zip(results.items(), axs):
#         ax.plot(sizes, times['optimized'], 'b-o', label='Optimized Implementation')
#         ax.plot(sizes, times['original'], 'r-o', label='Original Implementation')
#         ax.set_xlabel('Input Size')
#         ax.set_ylabel('Average Time (seconds)')
#         ax.set_title(f'Performance Comparison: MEC Implementations ({param_set})')
#         ax.legend()
#         ax.grid(True)
    
#     plt.tight_layout()
#     plt.savefig('comete_benchmark_results.png')
#     plt.close()

# def main():
#     # Test cases of different sizes
#     sizes = [10, 50, 100, 500, 1000, 5000]
#     n_runs = 100
    
#     # Different parameter combinations to test
#     alphas = [1.0, 1.2, 0.5, 1.5]
#     betas = [1.0, 1.2, 1.5, 0.5]
    
#     # Run benchmarks
#     print(f"Running benchmarks with {n_runs} iterations each...")
#     results = run_benchmarks(sizes, alphas, betas, n_runs)
    
#     # Plot results
#     plot_results(sizes, results)
    
#     # Print summary
#     print("\nSummary:")
#     for param_set, times in results.items():
#         print(f"\nFor {param_set}:")
#         for size, opt_t, orig_t in zip(sizes, times['optimized'], times['original']):
#             ratio = orig_t / opt_t
#             faster = "Optimized" if opt_t < orig_t else "Original"
#             print(f"Size {size:4d}: {faster} implementation is {abs(1-ratio)*100:.1f}% faster")

# if __name__ == "__main__":
#     main()
