import unittest
import numpy as np
from src.measures.metrics.literature.shannon import ShannonPol

class TestShannonPol(unittest.TestCase):
    def setUp(self):
        self.measure = ShannonPol()
        self.x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    def test_uniform_distribution(self):
        """Test ShannonPol with uniform distribution."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        result = self.measure.compute(self.x, weights)
        # Should be a positive value
        self.assertGreater(result, 0)
    
    def test_bimodal_distribution(self):
        """Test ShannonPol with bimodal distribution."""
        weights = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
        
        result = self.measure.compute(self.x, weights)
        # Bimodal should have higher polarization than uniform
        uniform_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        uniform_result = self.measure.compute(self.x, uniform_weights)
        
        self.assertGreater(result, uniform_result)
    
    def test_unimodal_distribution(self):
        """Test ShannonPol with unimodal distribution."""
        weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        
        result = self.measure.compute(self.x, weights)
        # Unimodal should have lower polarization than uniform
        uniform_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        uniform_result = self.measure.compute(self.x, uniform_weights)
        
        self.assertLess(result, uniform_result)
    
    def test_single_point_distribution(self):
        """Test ShannonPol with all mass at one point."""
        weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        
        result = self.measure.compute(self.x, weights)
        # Should be close to zero (minimum polarization)
        self.assertLess(result, 0.1)
    
    def test_increasing_polarization(self):
        """Test that polarization increases with more extreme distributions."""
        # Series of increasingly polarized distributions
        distributions = [
            np.array([0.0, 0.0, 1.0, 0.0, 0.0]),  # Fully concentrated (least polarized)
            np.array([0.1, 0.2, 0.4, 0.2, 0.1]),  # Unimodal
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),  # Uniform
            np.array([0.3, 0.1, 0.2, 0.1, 0.3]),  # Mild bimodal
            np.array([0.4, 0.1, 0.0, 0.1, 0.4]),  # Strong bimodal
            np.array([0.5, 0.0, 0.0, 0.0, 0.5])   # Perfect bimodal (most polarized)
        ]
        
        results = [self.measure.compute(self.x, w) for w in distributions]
        
        # Results should be monotonically increasing
        for i in range(1, len(results)):
            self.assertGreaterEqual(results[i], results[i-1])
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme distributions."""
        # Distribution with zeros
        weights = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
        
        # Should not raise error
        result = self.measure.compute(self.x, weights)
        self.assertTrue(np.isfinite(result))
        
        # Edge case: concentration with very small weights elsewhere
        weights = np.array([1e-10, 1e-10, 1.0-2e-10, 0.0, 0.0])
        
        # Should not overflow
        result = self.measure.compute(self.x, weights)
        self.assertTrue(np.isfinite(result))

if __name__ == "__main__":
    unittest.main()
