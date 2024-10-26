import numpy as np
from typing import Tuple
from src.metrics.literature.emd import EMDPolSciPy, EMDPol

def compare_implementations(x: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    """Compare results from both implementations."""
    emd_scipy = EMDPolSciPy()
    emd_original = EMDPol()
    
    result_scipy = emd_scipy(x, weights)
    result_original = emd_original(x, weights)
    
    return result_scipy, result_original

def generate_random_weights(size: int) -> np.ndarray:
    """Generate random weights that sum to 1."""
    weights = np.random.random(size)
    return weights / np.sum(weights)

def run_multiple_tests(n_tests: int = 1000, tolerance: float = 1e-10):
    """Run multiple random tests for different Likert scales."""
    scales = {
        "5-point": 5,
        "7-point": 7,
        "3-point": 3  # Agregado para completitud
    }
    
    results = {scale: {"max_diff": 0, "avg_diff": 0, "tests_failed": 0} 
              for scale in scales}
    
    for scale_name, n_points in scales.items():
        print(f"\nTesting {scale_name} Likert scale ({n_tests} iterations):")
        x = np.linspace(0, 1, n_points)
        
        differences = []
        for i in range(n_tests):
            weights = generate_random_weights(n_points)
            scipy_res, orig_res = compare_implementations(x, weights)
            diff = abs(scipy_res - orig_res)
            differences.append(diff)
            
            if diff > tolerance:
                results[scale_name]["tests_failed"] += 1
            
            if diff > results[scale_name]["max_diff"]:
                results[scale_name]["max_diff"] = diff
                results[scale_name]["worst_case_weights"] = weights
        
        results[scale_name]["avg_diff"] = np.mean(differences)
        
        # Print results for this scale
        print(f"Average difference: {results[scale_name]['avg_diff']:.2e}")
        print(f"Maximum difference: {results[scale_name]['max_diff']:.2e}")
        print(f"Tests failed (diff > {tolerance:.0e}): {results[scale_name]['tests_failed']}")
        
        if results[scale_name]["tests_failed"] > 0:
            print("\nWorst case scenario:")
            print("Weights:", results[scale_name]["worst_case_weights"])
            scipy_res, orig_res = compare_implementations(x, results[scale_name]["worst_case_weights"])
            print(f"SciPy result: {scipy_res}")
            print(f"Original result: {orig_res}")
    
    return results

def run_specific_cases():
    """Run specific test cases of interest."""
    print("\nTesting specific cases:")
    
    # Extreme case (all weight on endpoints)
    print("\nExtreme case (weights on endpoints):")
    x = np.linspace(0, 1, 5)
    w = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
    scipy_res, orig_res = compare_implementations(x, w)
    print(f"SciPy result: {scipy_res:.6f}")
    print(f"Original result: {orig_res:.6f}")
    print(f"Absolute difference: {abs(scipy_res - orig_res):.2e}")

    # Uniform distribution
    print("\nUniform distribution:")
    w = np.ones(5) / 5
    scipy_res, orig_res = compare_implementations(x, w)
    print(f"SciPy result: {scipy_res:.6f}")
    print(f"Original result: {orig_res:.6f}")
    print(f"Absolute difference: {abs(scipy_res - orig_res):.2e}")

if __name__ == "__main__":
    np.random.seed(42)  # Para reproducibilidad
    results = run_multiple_tests(n_tests=1000)
    run_specific_cases()
