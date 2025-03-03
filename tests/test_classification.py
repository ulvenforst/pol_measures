#!/usr/bin/env python3
"""
Simplified script to test the classification functionality.
This focuses on just basic classification without extensive formatting.
"""

import sys
import os
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.measures.metrics.proposed.mec import MECNormalized
    from src.measures.metrics.literature.esteban_ray import EstebanRay
    from src.measures.metrics.proposed.bipol import BiPol
    
    print("Successfully imported measures")
except ImportError as e:
    print(f"Import error: {e}")
    print("\nTry running this script from the project root directory.")
    sys.exit(1)

def main():
    # Create test data
    x = np.linspace(0, 1, 5)
    bimodal = np.array([0.4, 0.1, 0.4, 0.1, 0.0])
    
    print("\n===== TESTING ESTEBAN-RAY (alpha=0.8) =====")
    er_standard = EstebanRay(alpha=0.8)
    
    # Test raw value
    raw_value = er_standard(x, bimodal)
    print(f"Raw value: {raw_value:.4f}")
    
    # Test with k=3 classification
    try:
        result = er_standard(x, bimodal, labels=3)
        print(f"Classification (k=3): {result}")
    except Exception as e:
        print(f"Classification error: {str(e)}")
    
    # Test with non-standard alpha
    print("\n===== TESTING ESTEBAN-RAY (alpha=1.2) =====")
    er_nonstandard = EstebanRay(alpha=1.2)
    
    # Test raw value
    raw_value = er_nonstandard(x, bimodal)
    print(f"Raw value: {raw_value:.4f}")
    
    # Test with k=3 classification
    try:
        result = er_nonstandard(x, bimodal, labels=3)
        print(f"Classification (k=3): {result}")
    except Exception as e:
        print(f"Classification error: {str(e)}")
    
    print("\n===== TESTING MEC (alpha=2.0, beta=1.15) =====")
    mec_standard = MECNormalized(alpha=2.0, beta=1)
    
    # Test raw value
    raw_value = mec_standard(x, bimodal)
    print(f"Raw value: {raw_value:.4f}")
    
    # Test with k=3 classification
    try:
        result = mec_standard(x, bimodal, labels=3)
        print(f"Classification (k=3): {result}")
    except Exception as e:
        print(f"Classification error: {str(e)}")
    
    print("\n===== TESTING BIPOL =====")
    bipol = BiPol()
    
    # Test raw value
    raw_value = bipol(x, bimodal)
    print(f"Raw value: {raw_value:.4f}")
    
    # Test with k=3 classification
    try:
        result = bipol(x, bimodal, labels=3)
        print(f"Classification (k=3): {result}")
    except Exception as e:
        print(f"Classification error: {str(e)}")

if __name__ == "__main__":
    main()
