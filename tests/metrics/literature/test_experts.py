import unittest
import numpy as np
from src.measures.metrics.literature.experts import Experts

class TestExperts(unittest.TestCase):
    def setUp(self):
        self.measure = Experts()
        self.x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    def test_require_5_categories(self):
        """Test that the measure requires exactly 5 categories."""
        x_small = np.array([0.0, 0.5, 1.0])
        weights_small = np.array([0.3, 0.4, 0.3])
        
        with self.assertRaises(ValueError):
            self.measure.compute(x_small, weights_small)
        
        x_large = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        weights_large = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
        
        with self.assertRaises(ValueError):
            self.measure.compute(x_large, weights_large)
    
    def test_uniform_distribution(self):
        """Test Experts with uniform distribution."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        result = self.measure.compute(self.x, weights)
        self.assertTrue(0 <= result <= 1)
    
    def test_bimodal_distribution(self):
        """Test Experts with bimodal distribution."""
        weights = np.array([0.4, 0.1, 0.0, 0.1, 0.4])
        
        result = self.measure.compute(self.x, weights)
        # Bimodal should have higher polarization than uniform
        uniform_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        uniform_result = self.measure.compute(self.x, uniform_weights)
        
        self.assertGreater(result, uniform_result)
    
    def test_extreme_corners(self):
        """Test the effect of extreme corners on polarization."""
        # High weights on corners (n₁ and n₅)
        corner_weights = np.array([0.4, 0.1, 0.0, 0.1, 0.4])
        
        # High weights on second-level categories (n₂ and n₄)
        mid_weights = np.array([0.1, 0.4, 0.0, 0.4, 0.1])
        
        corner_result = self.measure.compute(self.x, corner_weights)
        mid_result = self.measure.compute(self.x, mid_weights)
        
        # Formula gives higher weight to n₁n₅ than n₂n₄
        self.assertGreater(corner_result, mid_result)
    
    def test_known_distributions(self):
        """Test with distributions that have known or expected results."""
        # Extreme bimodal (n₁ and n₅ only)
        extreme_bimodal = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
        
        # Uniform distribution
        uniform = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        # Central unimodal
        central_unimodal = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        
        extreme_result = self.measure.compute(self.x, extreme_bimodal)
        uniform_result = self.measure.compute(self.x, uniform)
        central_result = self.measure.compute(self.x, central_unimodal)
        
        # Extreme bimodal should have highest polarization
        self.assertGreater(extreme_result, uniform_result)
        self.assertGreater(extreme_result, central_result)
        
        # Central unimodal should have lowest polarization
        self.assertLess(central_result, uniform_result)

if __name__ == "__main__":
    unittest.main()
