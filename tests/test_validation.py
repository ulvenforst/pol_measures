import unittest
import numpy as np
from src.measures.validation import validate_histogram, minmax_normalize_x, validate_parameters

class TestValidation(unittest.TestCase):
    def test_minmax_normalize_x(self):
        """Test normalization of x values."""
        x = np.array([1, 3, 5, 7, 9])
        result = minmax_normalize_x(x)
        expected = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(result, expected)
        
        x = np.array([5, 5, 5])
        result = minmax_normalize_x(x)
        expected = np.array([0, 0, 0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_validate_histogram_valid(self):
        """Test validating a valid histogram."""
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        x_valid, w_valid = validate_histogram(x, weights)
        
        np.testing.assert_array_almost_equal(w_valid, weights)
        np.testing.assert_array_almost_equal(x_valid, x)
    
    def test_validate_histogram_error_shape(self):
        """Test validation fails when shapes don't match."""
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        weights = np.array([0.2, 0.2, 0.2, 0.2])
        
        with self.assertRaises(ValueError):
            validate_histogram(x, weights)
    
    def test_validate_histogram_error_min_points(self):
        """Test validation fails with too few points."""
        x = np.array([0.5])
        weights = np.array([1.0])
        
        with self.assertRaises(ValueError):
            validate_histogram(x, weights)
    
    def test_validate_histogram_error_not_increasing(self):
        """Test validation fails when x is not strictly increasing."""
        x = np.array([0.0, 0.25, 0.2, 0.75, 1.0])
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        
        with self.assertRaises(ValueError):
            validate_histogram(x, weights)
    
    def test_validate_histogram_error_negative_weights(self):
        """Test validation fails with negative weights."""
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        weights = np.array([0.2, -0.2, 0.4, 0.3, 0.3])
        
        with self.assertRaises(ValueError):
            validate_histogram(x, weights)
    
    def test_validate_histogram_error_all_zero_weights(self):
        """Test validation fails when all weights are zero."""
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        weights = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        
        with self.assertRaises(ValueError):
            validate_histogram(x, weights)
    
    def test_validate_parameters(self):
        """Test parameter validation."""
        validate_parameters(alpha=1.0, beta=2.0)
        
        with self.assertRaises(TypeError):
            validate_parameters(alpha="string")
        
        with self.assertRaises(ValueError):
            validate_parameters(alpha=-1.0)

if __name__ == "__main__":
    unittest.main()
