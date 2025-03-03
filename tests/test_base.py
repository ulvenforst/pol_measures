import unittest
import numpy as np
from src.measures.base import PolarizationMeasure, ParametricPolarizationMeasure
from src.measures.thresholds import THRESHOLDS

class MockPolarizationMeasure(PolarizationMeasure):
    """A simple mock implementation for testing the base class."""
    def compute(self, x, weights):
        return float(np.mean(x * weights))

class MockParametricPolarizationMeasure(ParametricPolarizationMeasure):
    """A mock implementation of a parametric polarization measure."""
    def compute(self, x, weights):
        return float(np.mean(x * weights) * self.parameters.get('factor', 1.0))

class TestPolarizationMeasure(unittest.TestCase):
    def setUp(self):
        self.measure = MockPolarizationMeasure()
        self.x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        self.weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    
    def test_call_return_value(self):
        """Test that __call__ returns the correct value when labels=None."""
        result = self.measure(self.x, self.weights)
        expected = self.measure.compute(self.x, self.weights)
        self.assertAlmostEqual(result, expected)
    
    def test_measure_id(self):
        """Test that measure_id returns the class name by default."""
        self.assertEqual(self.measure.measure_id, "MockPolarizationMeasure")
        
        # Test setter
        self.measure.measure_id = "custom_id"
        self.assertEqual(self.measure.measure_id, "custom_id")
    
    def test_has_parameter_sets(self):
        """Test _has_parameter_sets method."""
        # Mock a measure with parameter sets
        self.measure.measure_id = "EstebanRay"
        self.assertTrue(self.measure._has_parameter_sets())
        
        # Mock a measure without parameter sets
        self.measure.measure_id = "BiPol"
        self.assertFalse(self.measure._has_parameter_sets())

class TestParametricPolarizationMeasure(unittest.TestCase):
    def setUp(self):
        self.measure = MockParametricPolarizationMeasure(factor=2.0)
        self.x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        self.weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    
    def test_parameter_initialization(self):
        """Test that parameters are correctly initialized."""
        self.assertEqual(self.measure.parameters["factor"], 2.0)
    
    def test_update_parameters(self):
        """Test updating parameters."""
        self.measure.update_parameters(factor=3.0, new_param=1.0)
        self.assertEqual(self.measure.parameters["factor"], 3.0)
        self.assertEqual(self.measure.parameters["new_param"], 1.0)
        
    def test_get_parameters(self):
        """Test get_parameters returns a copy."""
        params = self.measure.get_parameters()
        self.assertEqual(params, self.measure.parameters)
        
        # Verify it's a copy
        params["factor"] = 5.0
        self.assertNotEqual(params["factor"], self.measure.parameters["factor"])
    
    def test_compute_with_parameters(self):
        """Test compute method uses parameters."""
        result = self.measure.compute(self.x, self.weights)
        expected = float(np.mean(self.x * self.weights) * 2.0)
        self.assertAlmostEqual(result, expected)

if __name__ == "__main__":
    unittest.main()
