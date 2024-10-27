# test_comete_correctness.py

import numpy as np
from typing import Tuple
from src.measures.metrics.proposed import Comete, CometeOptimized

def compare_implementations(x: np.ndarray, weights: np.ndarray, 
                          alpha: float = 1.0, beta: float = 1.0) -> Tuple[float, float]:
    """Compare results from both implementations."""
    comete_orig = Comete(alpha=alpha, beta=beta)
    comete_opt = CometeOptimized(alpha=alpha, beta=beta)
    
    result_orig = comete_orig(x, weights)
    result_opt = comete_opt(x, weights)
    
    return result_orig, result_opt

def generate_random_weights(size: int) -> np.ndarray:
    """Generate random weights that sum to 1."""
    weights = np.random.random(size)
    return weights / np.sum(weights)

def run_multiple_tests(n_tests: int = 1000, tolerance: float = 1e-10):
    """Run multiple random tests for different Likert scales and parameter values."""
    scales = {
        "5-point": 5,
        "7-point": 7,
        "3-point": 3
    }
    
    parameter_sets = [
        (1.0, 1.0),    # default
        (1.2, 1.2),    # higher values
        (0.5, 1.5),    # different alpha/beta
        (1.5, 0.5),    # different beta/alpha
    ]
    
    results = {
        f"{scale}_{alpha}_{beta}": {
            "max_diff": 0, 
            "avg_diff": 0, 
            "tests_failed": 0
        } 
        for scale in scales 
        for alpha, beta in parameter_sets
    }
    
    for scale_name, n_points in scales.items():
        x = np.linspace(0, 1, n_points)
        
        for alpha, beta in parameter_sets:
            key = f"{scale_name}_{alpha}_{beta}"
            print(f"\nTesting {scale_name} Likert scale with α={alpha}, β={beta} ({n_tests} iterations):")
            
            differences = []
            for i in range(n_tests):
                weights = generate_random_weights(n_points)
                orig_res, opt_res = compare_implementations(x, weights, alpha, beta)
                diff = abs(orig_res - opt_res)
                differences.append(diff)
                
                if diff > tolerance:
                    results[key]["tests_failed"] += 1
                
                if diff > results[key]["max_diff"]:
                    results[key]["max_diff"] = diff
                    results[key]["worst_case_weights"] = weights
                    results[key]["worst_case_params"] = (alpha, beta)
            
            results[key]["avg_diff"] = np.mean(differences)
            
            # Print results
            print(f"Average difference: {results[key]['avg_diff']:.2e}")
            print(f"Maximum difference: {results[key]['max_diff']:.2e}")
            print(f"Tests failed (diff > {tolerance:.0e}): {results[key]['tests_failed']}")
            
            if results[key]["tests_failed"] > 0:
                print("\nWorst case scenario:")
                print("Weights:", results[key]["worst_case_weights"])
                print(f"Parameters (α,β): {results[key]['worst_case_params']}")
                orig_res, opt_res = compare_implementations(
                    x, 
                    results[key]["worst_case_weights"],
                    *results[key]["worst_case_params"]
                )
                print(f"Original result: {orig_res}")
                print(f"Optimized result: {opt_res}")
    
    return results

def run_specific_cases():
    """Run specific test cases of interest."""
    print("\nTesting specific cases:")
    
    # Definir casos de prueba
    test_cases = {
        "Bimodal": np.array([0.5, 0.0, 0.0, 0.0, 0.5]),
        "Uniform": np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        "Unimodal": np.array([0.1, 0.2, 0.4, 0.2, 0.1]),
        "Left skewed": np.array([0.5, 0.3, 0.1, 0.1, 0.0]),
        "Right skewed": np.array([0.0, 0.1, 0.1, 0.3, 0.5]),
        "Single spike": np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    }
    
    x = np.linspace(0, 1, 5)
    
    for case_name, weights in test_cases.items():
        print(f"\n{case_name} distribution:")
        orig_res, opt_res = compare_implementations(x, weights)
        print(f"Original result: {orig_res:.6f}")
        print(f"Optimized result: {opt_res:.6f}")
        print(f"Absolute difference: {abs(orig_res - opt_res):.2e}")
        
        # Probar también con diferentes parámetros
        alpha, beta = 1.2, 1.2
        print(f"\nWith α=β={alpha}:")
        orig_res, opt_res = compare_implementations(x, weights, alpha, beta)
        print(f"Original result: {orig_res:.6f}")
        print(f"Optimized result: {opt_res:.6f}")
        print(f"Absolute difference: {abs(orig_res - opt_res):.2e}")

if __name__ == "__main__":
    np.random.seed(42)  # Para reproducibilidad
    results = run_multiple_tests(n_tests=1000)
    run_specific_cases()
