import unittest
import numpy as np
from src.measures.metrics.proposed.mec import MEC, MECNormalized

class TestMEC(unittest.TestCase):
    def setUp(self):
        self.measure = MEC()
        self.x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    def test_parameter_initialization(self):
        """Test parameter initialization."""
        self.assertEqual(self.measure.parameters["alpha"], 2)
        self.assertEqual(self.measure.parameters["beta"], 1.15)
        
        # Custom parameters
        custom_measure = MEC(alpha=1.5, beta=1.0)
        self.assertEqual(custom_measure.parameters["alpha"], 1.5)
        self.assertEqual(custom_measure.parameters["beta"], 1.0)
    
    def test_uniform_distribution(self):
        """Test MEC with uniform distribution."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        result = self.measure.compute(self.x, weights)
        # Should be a positive value
        self.assertGreater(result, 0)
    
    def test_bimodal_distribution(self):
        """Test MEC with bimodal distribution."""
        weights = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
        
        result = self.measure.compute(self.x, weights)
        # Bimodal should give higher value than uniform
        uniform_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        uniform_result = self.measure.compute(self.x, uniform_weights)
        
        self.assertGreater(result, uniform_result)
    
    def test_single_point_distribution(self):
        """Test MEC with all mass at one point."""
        weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        
        result = self.measure.compute(self.x, weights)
        # Should be close to zero (all consensus at one point)
        self.assertLess(result, 0.1)
    
    def test_parameter_effect(self):
        """Test how different parameter values affect the result."""
        weights = np.array([0.4, 0.1, 0.0, 0.1, 0.4])
        
        # Alpha affects the weight of different population segments
        mec1 = MEC(alpha=1.0, beta=1.15)
        mec2 = MEC(alpha=2.0, beta=1.15)
        
        result1 = mec1.compute(self.x, weights)
        result2 = mec2.compute(self.x, weights)
        
        # Results should be different
        self.assertNotEqual(result1, result2)
        
        # Beta affects the distance metric
        mec3 = MEC(alpha=2.0, beta=1.0)
        mec4 = MEC(alpha=2.0, beta=1.5)
        
        result3 = mec3.compute(self.x, weights)
        result4 = mec4.compute(self.x, weights)
        
        # Results should be different
        self.assertNotEqual(result3, result4)

class TestMECNormalized(unittest.TestCase):
    def setUp(self):
        self.measure = MECNormalized()
        self.x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    def test_get_max_distribution(self):
        """Test the _get_max_distribution method."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        x_max, w_max = self.measure._get_max_distribution(self.x, weights)
        
        # Should return endpoints and 0.5 mass at each
        np.testing.assert_array_almost_equal(x_max, np.array([0.0, 1.0]))
        np.testing.assert_array_almost_equal(w_max, np.array([0.5, 0.5]))
    
    def test_uniform_distribution(self):
        """Test normalized MEC with uniform distribution."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        result = self.measure.compute(self.x, weights)
        # Should be in [0,1] range
        self.assertTrue(0 <= result <= 1)
    
    def test_bimodal_distribution(self):
        """Test normalized MEC with bimodal distribution."""
        weights = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
        
        result = self.measure.compute(self.x, weights)
        # Should be close to 1 (maximum polarization)
        self.assertGreater(result, 0.9)
    
    def test_single_point_distribution(self):
        """Test normalized MEC with all mass at one point."""
        weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        
        result = self.measure.compute(self.x, weights)
        # Should be close to 0 (minimum polarization)
        self.assertLess(result, 0.1)
    
    def test_normalization_effect(self):
        """Test that normalization produces values in [0,1]."""
        # Generate various distributions
        distributions = [
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),  # Uniform
            np.array([0.5, 0.0, 0.0, 0.0, 0.5]),  # Perfect bimodal
            np.array([0.1, 0.2, 0.4, 0.2, 0.1]),  # Unimodal center
            np.array([0.4, 0.3, 0.2, 0.1, 0.0]),  # Asymmetric
            np.array([0.0, 0.0, 0.0, 0.0, 1.0])   # Single point
        ]
        
        for weights in distributions:
            result = self.measure.compute(self.x, weights)
            self.assertTrue(0 <= result <= 1)

if __name__ == "__main__":
    unittest.main()
