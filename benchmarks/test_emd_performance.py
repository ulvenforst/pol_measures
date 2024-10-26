import numpy as np
import time
from typing import Callable, List, Tuple
import matplotlib.pyplot as plt
from src.metrics.literature.emd import EMDPolSciPy, EMDPol

def generate_test_case(size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate test case of specified size."""
    x = np.linspace(0, 1, size)
    weights = np.random.random(size)
    weights /= np.sum(weights)
    return x, weights

def benchmark_function(func: Callable, x: np.ndarray, weights: np.ndarray, 
                      n_runs: int = 100) -> float:
    """Benchmark a function with given inputs."""
    start_time = time.perf_counter()
    for _ in range(n_runs):
        func(x, weights)
    end_time = time.perf_counter()
    return (end_time - start_time) / n_runs

def run_benchmarks(sizes: List[int], n_runs: int = 100) -> Tuple[List[float], List[float]]:
    """Run benchmarks for both implementations with different input sizes."""
    scipy_times = []
    original_times = []
    
    emd_scipy = EMDPolSciPy()
    emd_original = EMDPol()
    
    for size in sizes:
        print(f"Testing size {size}...")
        x, weights = generate_test_case(size)
        
        scipy_time = benchmark_function(emd_scipy, x, weights, n_runs)
        original_time = benchmark_function(emd_original, x, weights, n_runs)
        
        scipy_times.append(scipy_time)
        original_times.append(original_time)
        
        print(f"SciPy implementation: {scipy_time:.6f} seconds")
        print(f"Original implementation: {original_time:.6f} seconds")
        print("-" * 50)
    
    return scipy_times, original_times

def plot_results(sizes: List[int], scipy_times: List[float], 
                original_times: List[float]) -> None:
    """Plot benchmark results."""
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, scipy_times, 'b-o', label='SciPy Implementation')
    plt.plot(sizes, original_times, 'r-o', label='Original Implementation')
    plt.xlabel('Input Size')
    plt.ylabel('Average Time (seconds)')
    plt.title('Performance Comparison: EMD Implementations')
    plt.legend()
    plt.grid(True)
    plt.savefig('emd_benchmark_results.png')
    plt.close()

def main():
    # Test cases of different sizes
    sizes = [10, 50, 100, 500, 1000, 5000]
    n_runs = 100
    
    # Run benchmarks
    print(f"Running benchmarks with {n_runs} iterations each...")
    scipy_times, original_times = run_benchmarks(sizes, n_runs)
    
    # Plot results
    plot_results(sizes, scipy_times, original_times)
    
    # Print summary
    print("\nSummary:")
    for size, scipy_t, orig_t in zip(sizes, scipy_times, original_times):
        ratio = orig_t / scipy_t
        faster = "SciPy" if scipy_t < orig_t else "Original"
        print(f"Size {size:4d}: {faster} implementation is {abs(1-ratio)*100:.1f}% faster")

if __name__ == "__main__":
    main()
