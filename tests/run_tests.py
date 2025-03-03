#!/usr/bin/env python
import unittest
import os
import sys

# Asegurar que podamos importar el paquete measures
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_all_tests():
    """Discover and run all tests in the tests directory."""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=os.path.dirname(__file__), pattern='test_*.py')
    
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
