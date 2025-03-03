import unittest
import numpy as np
from typing import Dict, Any, Tuple, cast, List, Optional
from src.measures.base import PolarizationMeasure, ParametricPolarizationMeasure
from src.measures.thresholds import THRESHOLDS, CATEGORY_LABELS

class MockMeasureWithConfigurableValue(PolarizationMeasure):
    """A mock measure that returns a configurable value for testing."""
    
    def __init__(self, fixed_value: float = 0.5):
        super().__init__()
        self.fixed_value = fixed_value
        self._mock_parameters = {}
    
    def compute(self, x, weights):
        return float(self.fixed_value)
    
    def find_matching_parameter_set(self) -> Optional[str]:
        """Mock implementation that ensures tests don't get 'no_classification'."""
        if self.measure_id not in THRESHOLDS:
            return None
        
        if "_params" in THRESHOLDS[self.measure_id]:
            return "default"
        
        return "default"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return simulated parameters for the current measure_id."""
        if self.measure_id in THRESHOLDS and "_params" in THRESHOLDS[self.measure_id]:
            # Return the default parameters for this measure
            return THRESHOLDS[self.measure_id]["_params"]["default"].copy()
        return {}

class MockParametricMeasure(ParametricPolarizationMeasure):
    """A mock parametric measure for testing parameter-based functionality."""
    
    def __init__(self, fixed_value: float = 0.5, **parameters):
        super().__init__(**parameters)
        self.fixed_value = fixed_value
    
    def compute(self, x, weights):
        return float(self.fixed_value)

class TestThresholdsComprehensive(unittest.TestCase):
    """Comprehensive tests for all measures and classification schemes in THRESHOLDS."""
    
    def setUp(self):
        self.measure = MockMeasureWithConfigurableValue()
        self.x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        self.weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    
    def test_thresholds_match_expected_values(self):
        """Verify that the thresholds in THRESHOLDS match the expected values."""
        self.assertEqual(THRESHOLDS["EstebanRay"]["default"]["kmeans"][3], [0.3791, 0.5141])
        self.assertEqual(THRESHOLDS["EstebanRay"]["default"]["kmeans"][4], [0.3501, 0.4467, 0.5660])
        self.assertEqual(THRESHOLDS["EstebanRay"]["default"]["kmeans"][5], [0.3196, 0.3973, 0.4809, 0.5921])
        
        self.assertEqual(THRESHOLDS["EstebanRay"]["default"]["percentile"][3], [0.3609, 0.4466])  # 33rd/66th
        self.assertEqual(THRESHOLDS["EstebanRay"]["default"]["percentile"][4], [0.3407, 0.4031, 0.4761])  # 25th/50th/75th
        self.assertEqual(THRESHOLDS["EstebanRay"]["default"]["percentile"][5], [0.3275, 0.3781, 0.4295, 0.4965])  # 20th/40th/60th/80th
        
        self.assertEqual(THRESHOLDS["BiPol"]["kmeans"][3], [0.4525, 0.6267])
        self.assertEqual(THRESHOLDS["BiPol"]["kmeans"][4], [0.3940, 0.5317, 0.6702])
        self.assertEqual(THRESHOLDS["BiPol"]["kmeans"][5], [0.3543, 0.4730, 0.5816, 0.7024])
        
        self.assertEqual(THRESHOLDS["BiPol"]["percentile"][3], [0.4726, 0.5994])  # 33rd/66th
        self.assertEqual(THRESHOLDS["BiPol"]["percentile"][4], [0.4380, 0.5376, 0.6383])  # 25th/50th/75th
        self.assertEqual(THRESHOLDS["BiPol"]["percentile"][5], [0.4140, 0.5000, 0.5756, 0.6627])  # 20th/40th/60th/80th
        
        self.assertEqual(THRESHOLDS["MECNormalized"]["default"]["kmeans"][3], [0.2233, 0.3801])
        self.assertEqual(THRESHOLDS["MECNormalized"]["default"]["kmeans"][4], [0.1718, 0.2841, 0.4271])
        self.assertEqual(THRESHOLDS["MECNormalized"]["default"]["kmeans"][5], [0.1513, 0.2421, 0.3369, 0.4689])
        
        self.assertEqual(THRESHOLDS["MECNormalized"]["default"]["percentile"][3], [0.1971, 0.2922])  # 33rd/66th
        self.assertEqual(THRESHOLDS["MECNormalized"]["default"]["percentile"][4], [0.1747, 0.2408, 0.3274])  # 25th/50th/75th
        self.assertEqual(THRESHOLDS["MECNormalized"]["default"]["percentile"][5], [0.1581, 0.2141, 0.2716, 0.3503]) # 20th/40th/60th/80th
    
    def test_all_thresholds_structure(self):
        """Test the overall structure of THRESHOLDS for all measures."""
        self.assertIn("EstebanRay", THRESHOLDS)
        self.assertIn("BiPol", THRESHOLDS)
        self.assertIn("MECNormalized", THRESHOLDS)
        
        for measure_name, measure_data in THRESHOLDS.items():
            with self.subTest(measure=measure_name):
                has_param_sets = "_params" in measure_data
                
                if has_param_sets:
                    self.assertIn("_params", measure_data)
                    self.assertIsInstance(measure_data["_params"], dict)
                    
                    self.assertIn("default", measure_data["_params"])
                    
                    for param_set, params in measure_data["_params"].items():
                        self.assertIsInstance(params, dict)
                        self.assertGreater(len(params), 0)
                    
                    for key in measure_data:
                        if key != "_params":
                            self.assertIn("kmeans", measure_data[key])
                            self.assertIn("percentile", measure_data[key])
                else:
                    self.assertIn("kmeans", measure_data)
                    self.assertIn("percentile", measure_data)
    
    def test_measure_classification_ranges(self):
        """Test that classifications match the expected ranges for all measures."""
        for measure_name in THRESHOLDS:
            with self.subTest(measure=measure_name):
                self.measure.measure_id = measure_name
                
                has_param_sets = "_params" in THRESHOLDS[measure_name]
                
                param_set = "default"
                
                if has_param_sets:
                    methods_data = THRESHOLDS[measure_name][param_set]
                else:
                    methods_data = THRESHOLDS[measure_name]
                
                for method in ["kmeans", "percentile"]:
                    method_data = methods_data[method]
                    
                    for num_cats, thresholds in method_data.items():
                        ranges = self._create_ranges(thresholds)
                        
                        for i, (lower, upper) in enumerate(ranges):
                            test_value = (lower + upper) / 2
                            
                            self.measure.fixed_value = test_value
                            
                            category = self.measure._classify_value(
                                test_value, num_cats, method, param_set
                            )
                            
                            expected_category = CATEGORY_LABELS[num_cats][i]
                            
                            self.assertEqual(
                                category, 
                                expected_category,
                                f"Value {test_value} in range [{lower:.4f}, {upper:.4f}] should be classified as {expected_category} "
                                f"but was {category} for {measure_name}, {method}, {num_cats} categories"
                            )
    
    def test_all_measures_call_with_labels(self):
        """Test __call__ with labels for all measures."""
        test_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for measure_name in THRESHOLDS:
            with self.subTest(measure=measure_name):
                self.measure.measure_id = measure_name
                
                if "_params" in THRESHOLDS[measure_name]:
                    default_params = THRESHOLDS[measure_name]["_params"]["default"]
                    
                    self.measure._mock_parameters = default_params.copy()
                
                for test_value in test_values:
                    self.measure.fixed_value = test_value
                    
                    result = self.measure(self.x, self.weights, labels=3)
                    
                    self.assertIsInstance(result, tuple)
                    
                    result_tuple = cast(Tuple[float, str], result)
                    value, category = result_tuple
                    
                    self.assertEqual(value, test_value)
                    
                    self.assertNotEqual(category, "no_classification", 
                                       f"Got 'no_classification' for measure {measure_name} with value {test_value}")
                    self.assertIn(category, CATEGORY_LABELS[3])
                    
                    result_all = self.measure(self.x, self.weights, labels="all")
                    self.assertIsInstance(result_all, dict)
                    
                    result_dict = cast(Dict[str, Any], result_all)
                    self.assertEqual(result_dict["value"], test_value)
                    self.assertIn("classifications", result_dict)
                    
                    # Check that each method and category count has a classification
                    classifications = result_dict["classifications"]
                    self.assertIn("kmeans", classifications)
                    self.assertIn("percentile", classifications)
    
    def _create_ranges(self, thresholds: List[float]) -> List[Tuple[float, float]]:
        """
        Create a list of ranges from threshold values.
        
        Args:
            thresholds: List of threshold values
            
        Returns:
            List of (lower, upper) bounds for each category
        """
        ranges = []
        
        ranges.append((0.0, thresholds[0]))
        
        for i in range(len(thresholds) - 1):
            ranges.append((thresholds[i], thresholds[i+1]))
        
        ranges.append((thresholds[-1], 1.0))
        
        return ranges
    
    def test_boundary_values(self):
        """Test values exactly at the boundaries of categories."""
        for measure_name in THRESHOLDS:
            with self.subTest(measure=measure_name):
                self.measure.measure_id = measure_name
                
                param_set = "default"
                
                has_param_sets = "_params" in THRESHOLDS[measure_name]
                
                if has_param_sets:
                    methods_data = THRESHOLDS[measure_name][param_set]
                else:
                    methods_data = THRESHOLDS[measure_name]
                
                if 3 in methods_data["kmeans"]:
                    thresholds = methods_data["kmeans"][3]
                    
                    for i, threshold in enumerate(thresholds):
                        self.measure.fixed_value = threshold
                        
                        # First threshold should classify as the second category (medium)
                        # because the rule is "value < threshold" for the first category
                        category = self.measure._classify_value(
                            threshold, 3, "kmeans", param_set
                        )
                        
                        # For the value exactly at the threshold, it should be classified
                        # in the category after the threshold
                        expected_category = CATEGORY_LABELS[3][i+1] if i < len(thresholds) else CATEGORY_LABELS[3][-1]
                        
                        self.assertEqual(
                            category, 
                            expected_category,
                            f"Value exactly at threshold {threshold} should be classified as {expected_category}"
                        )
    
    def test_each_threshold_value(self):
        """Test every threshold value for each measure to ensure correct ranges."""
        for measure_name in THRESHOLDS:
            with self.subTest(measure=measure_name):
                self.measure.measure_id = measure_name
                
                param_set = "default"
                
                if "_params" in THRESHOLDS[measure_name]:
                    threshold_data = THRESHOLDS[measure_name][param_set]
                else:
                    threshold_data = THRESHOLDS[measure_name]
                
                for method in ["kmeans", "percentile"]:
                    for num_cats, thresholds in threshold_data[method].items():
                        # Test values slightly below each threshold
                        for i, threshold in enumerate(thresholds):
                            # Value just below the threshold (should be in category i)
                            below_value = threshold - 1e-10
                            self.measure.fixed_value = below_value
                            
                            below_category = self.measure._classify_value(
                                below_value, num_cats, method, param_set
                            )
                            
                            expected_below_category = CATEGORY_LABELS[num_cats][i]
                            self.assertEqual(
                                below_category, 
                                expected_below_category,
                                f"Value {below_value} (just below threshold {threshold}) should be {expected_below_category}"
                            )
                            
                            above_value = threshold + 1e-10
                            self.measure.fixed_value = above_value
                            
                            above_category = self.measure._classify_value(
                                above_value, num_cats, method, param_set
                            )
                            
                            expected_above_category = CATEGORY_LABELS[num_cats][i+1]
                            self.assertEqual(
                                above_category, 
                                expected_above_category,
                                f"Value {above_value} (just above threshold {threshold}) should be {expected_above_category}"
                            )
    
    def test_parameter_set_matching(self):
        """Test that measures with parameter sets can find matching parameter sets."""
        for measure_name, measure_data in THRESHOLDS.items():
            if "_params" in measure_data:
                with self.subTest(measure=measure_name):
                    default_params = measure_data["_params"]["default"]
                    
                    param_measure = MockParametricMeasure(**default_params)
                    param_measure.measure_id = measure_name
                    
                    param_set = param_measure.find_matching_parameter_set()
                    self.assertEqual(param_set, "default")
                    
                    modified_params = default_params.copy()
                    for key in modified_params:
                        if isinstance(modified_params[key], (int, float)):
                            modified_params[key] += 1.0
                            break
                    
                    param_measure_modified = MockParametricMeasure(**modified_params)
                    param_measure_modified.measure_id = measure_name
                    
                    # Should not find matching parameter set
                    param_set = param_measure_modified.find_matching_parameter_set()
                    self.assertIsNone(param_set)

if __name__ == "__main__":
    unittest.main()
