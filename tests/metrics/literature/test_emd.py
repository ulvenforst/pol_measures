import unittest
import numpy as np
from src.measures.metrics.literature.emd import EMDPol, EMDPolSciPy

class TestEMDPol(unittest.TestCase):
    def setUp(self):
        self.measure = EMDPol()
        self.x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    def test_uniform_distribution(self):
        """Test EMDPol with uniform distribution."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        result = self.measure.compute(self.x, weights)
        self.assertTrue(0 < result < 0.5)
    
    def test_perfect_bimodal(self):
        """Test EMDPol with perfect bimodal distribution."""
        weights = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
        
        result = self.measure.compute(self.x, weights)
        self.assertAlmostEqual(result, 0.5, places=6)
    
    def test_unimodal_center(self):
        """Test EMDPol with unimodal centered distribution."""
        weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        
        result = self.measure.compute(self.x, weights)
        self.assertLess(result, 0.3)
    
    def test_boundary_case(self):
        """Test with weights at one extreme."""
        weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        
        result = self.measure.compute(self.x, weights)
        self.assertLess(result, 0.3)

class TestEMDPolSciPy(unittest.TestCase):
    def setUp(self):
        self.measure = EMDPolSciPy()
        self.x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    
    def test_create_target_distribution(self):
        """Test the target distribution creation."""
        target = self.measure._create_target_distribution(5)
        expected = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
        
        np.testing.assert_array_equal(target, expected)
    
    def test_uniform_distribution(self):
        """Test EMDPolSciPy with uniform distribution."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        result = self.measure.compute(self.x, weights)
        # Should be a moderate value between 0 and 0.5
        self.assertTrue(0 < result < 0.5)
    
    def test_perfect_bimodal(self):
        """Test EMDPolSciPy with perfect bimodal distribution."""
        weights = np.array([0.5, 0.0, 0.0, 0.0, 0.5])
        
        result = self.measure.compute(self.x, weights)
        self.assertAlmostEqual(result, 0.5, places=6)
    
    def test_unimodal_center(self):
        """Test EMDPolSciPy with unimodal centered distribution."""
        weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        
        result = self.measure.compute(self.x, weights)
        self.assertLess(result, 0.3)
    
    def test_consistency_with_emdpol(self):
        """Test consistency between EMDPol and EMDPolSciPy."""
        original = EMDPol()
        scipy_version = EMDPolSciPy()
        
        distributions = [
            np.array([0.2, 0.2, 0.2, 0.2, 0.2]),  # Uniform
            np.array([0.5, 0.0, 0.0, 0.0, 0.5]),  # Perfect bimodal
            np.array([0.0, 0.0, 1.0, 0.0, 0.0]),  # Unimodal center
        ]
        
        for weights in distributions:
            result1 = original.compute(self.x, weights)
            result2 = scipy_version.compute(self.x, weights)
            self.assertAlmostEqual(result1, result2, places=4)

if __name__ == "__main__":
    unittest.main()
