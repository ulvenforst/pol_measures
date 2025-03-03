import unittest
import numpy as np
from src.measures.metrics.literature.esteban_ray import EstebanRay

class TestEstebanRay(unittest.TestCase):
    def setUp(self):
        self.measure = EstebanRay()
        self.x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        self.assertEqual(self.measure.parameters["alpha"], 0.8)
        self.assertIsNone(self.measure.parameters["K"])
        
        custom_measure = EstebanRay(alpha=1.0, K=2.0)
        self.assertEqual(custom_measure.parameters["alpha"], 1.0)
        self.assertEqual(custom_measure.parameters["K"], 2.0)
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        EstebanRay(alpha=1.0)
        
        with self.assertRaises(ValueError):
            EstebanRay(alpha=0)
        
        with self.assertRaises(ValueError):
            EstebanRay(alpha=1.7)
    
    def test_uniform_distribution(self):
        """Test with uniform distribution."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        result = self.measure.compute(self.x, weights)
        self.assertTrue(0 < result < 1)
    
    def test_bimodal_distribution(self):
        """Test with bimodal distribution."""
        weights = np.array([0.4, 0.1, 0.0, 0.1, 0.4])
        
        result = self.measure.compute(self.x, weights)
        uniform_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        uniform_result = self.measure.compute(self.x, uniform_weights)
        
        self.assertGreater(result, uniform_result)
    
    def test_alpha_effect(self):
        """Test how alpha parameter affects results."""
        weights = np.array([0.4, 0.1, 0.0, 0.1, 0.4])
        
        er1 = EstebanRay(alpha=0.5)
        er2 = EstebanRay(alpha=1.0)
        er3 = EstebanRay(alpha=1.5)
        
        result1 = er1.compute(self.x, weights)
        result2 = er2.compute(self.x, weights)
        result3 = er3.compute(self.x, weights)
        
        self.assertNotEqual(result1, result2)
        self.assertNotEqual(result2, result3)
    
    def test_k_parameter(self):
        """Test the effect of K parameter."""
        weights = np.array([0.4, 0.1, 0.0, 0.1, 0.4])
        
        alpha = 0.8
        default_K = 1 / (2 * ((0.5) ** (2 + alpha)))
        
        default_measure = EstebanRay(alpha=alpha)
        default_result = default_measure.compute(self.x, weights)
        
        explicit_K_measure = EstebanRay(alpha=alpha, K=default_K)
        explicit_K_result = explicit_K_measure.compute(self.x, weights)
        
        self.assertAlmostEqual(default_result, explicit_K_result, places=6)
        
        double_K_measure = EstebanRay(alpha=alpha, K=2*default_K)
        double_K_result = double_K_measure.compute(self.x, weights)
        
        self.assertAlmostEqual(double_K_result / default_result, 2.0, places=6)
        
        fixed_K_measure = EstebanRay(alpha=alpha, K=3.0)
        fixed_K_result = fixed_K_measure.compute(self.x, weights)
        
        expected_ratio = 3.0 / default_K
        self.assertAlmostEqual(fixed_K_result / default_result, expected_ratio, places=6)
    
    def test_find_matching_parameter_set(self):
        """Test finding matching parameter set in thresholds."""
        param_set = self.measure.find_matching_parameter_set()
        self.assertEqual(param_set, "default")
        
        custom_measure = EstebanRay(alpha=1.0)
        param_set = custom_measure.find_matching_parameter_set()
        self.assertIsNone(param_set)

if __name__ == "__main__":
    unittest.main()
