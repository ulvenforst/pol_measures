from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, Dict, Any, List
import numpy as np
import math
from .validation import validate_histogram
from .thresholds import THRESHOLDS, CATEGORY_LABELS

class PolarizationMeasure(ABC):
    """Base class for all polarization measures."""

    def __init__(self) -> None:
        self._cached_result: Optional[float] = None
        self._measure_id: Optional[str] = None

    @property
    def measure_id(self) -> str:
        """Return the identifier used for threshold lookup."""
        if self._measure_id is None:
            self._measure_id = self.__class__.__name__
        return self._measure_id
    
    @measure_id.setter
    def measure_id(self, value: str) -> None:
        """Set a custom identifier for threshold lookup."""
        self._measure_id = value

    @abstractmethod
    def compute(self, x: np.ndarray, weights: np.ndarray) -> float:
        """Compute the polarization measure."""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Return the parameters used by this measure.
        By default, returns an empty dict for non-parametric measures.
        Parametric measures should override this method.
        """
        return {}
    
    def _has_parameter_sets(self) -> bool:
        """Check if this measure uses multiple parameter sets."""
        return (self.measure_id in THRESHOLDS and 
                "_params" in THRESHOLDS[self.measure_id])
    
    def find_matching_parameter_set(self) -> Optional[str]:
        """
        Find a parameter set in THRESHOLDS that matches the current parameters.
        Returns the parameter set key if found, None otherwise.
        Compatible with both old and new implementations, with type-safe access.
        """
        if self.measure_id not in THRESHOLDS:
            return None
            
        # Get current parameters (type-safe version)
        try:
            if hasattr(self, 'get_parameters') and callable(getattr(self, 'get_parameters')):
                current_params = self.get_parameters()
            elif hasattr(self, 'parameters'):
                params = getattr(self, 'parameters', {})
                current_params = params.copy() if params else {}
            else:
                current_params = {}
        except Exception:
            current_params = {}

        FLOAT_TOLERANCE = 1e-9
        # Check if the measure uses parameter sets
        if self._has_parameter_sets():
            # Multiple parameter sets are defined
            param_sets = THRESHOLDS[self.measure_id]["_params"]
            
            for param_set_key, default_params in param_sets.items():
                # Check if critical parameters match (only check parameters in default_params)
                all_match = True
                for param, default_value in default_params.items():
                    if param not in current_params:
                        all_match = False
                        break
                    
                    # Compare parameter values with tolerance for floating point
                    if isinstance(default_value, (int, float)) and isinstance(current_params[param], (int, float)):
                        if not math.isclose(default_value, current_params[param], rel_tol=FLOAT_TOLERANCE):
                            all_match = False
                            break
                    elif default_value != current_params[param]:
                        all_match = False
                        break
                
                # If all default parameters match, consider it a match (ignoring extra params)
                if all_match:
                    return param_set_key
            
            # No matching parameter set found
            return None
        else:
            # No parameter sets defined, assume default behavior
            return "default" if not current_params else None

    def __call__(
        self, 
        x: np.ndarray, 
        weights: np.ndarray, 
        labels: Optional[Union[int, str]] = None,
        method: str = "kmeans"
    ) -> Union[float, Tuple[float, str], Dict[str, Any]]:
        """
        Compute polarization and optionally classify the result.
        
        Args:
            x: The positions of the distribution
            weights: The weights of the distribution
            labels: 
                - None: Return only the numerical value (default)
                - int: Return value and classification using k clusters
                - "all": Return all available classification schemes
            method: Classification method ("kmeans" or "percentile")
            
        Returns:
            - float: When labels=None
            - (float, str): When labels is an integer
            - dict: When labels="all"
        """
        x, weights = validate_histogram(x, weights)
        self._cached_result = self.compute(x, weights)

        
        if labels is None:
            return self._cached_result
        
        # Try to find matching parameter set
        param_set = self.find_matching_parameter_set()
        
        # Handle classifications
        if labels == "all":
            if param_set is None:
                return {
                    "value": self._cached_result, 
                    "classifications": {}, 
                    "error": "No matching thresholds found for the current parameters"
                }
            return self._get_all_classifications(self._cached_result, param_set)
        
        if isinstance(labels, int) and labels >= 2:
            if param_set is None:
                return self._cached_result, "no_classification"
            
            try:
                category = self._classify_value(self._cached_result, num_categories=labels, 
                                               method=method, param_set=param_set)
                return self._cached_result, category
            except ValueError:
                # No thresholds for this specific combination
                return self._cached_result, "no_classification"
        
        raise ValueError(f"Invalid value for 'labels': {labels}. Must be None, a positive integer, or 'all'.")
    
    def _get_thresholds(self, num_categories: int, method: str = "kmeans", param_set: str = "default") -> List[float]:
        """
        Get thresholds for this measure from the threshold database.
        
        Args:
            num_categories: Number of categories for classification
            method: Classification method (kmeans or percentile)
            param_set: Parameter set to use for threshold lookup
            
        Returns:
            List of threshold values
            
        Raises:
            ValueError: If thresholds cannot be found
        """
        if self.measure_id not in THRESHOLDS:
            raise ValueError(f"No thresholds available for measure: {self.measure_id}")
        
        # Check if we need to use a parameter set
        thresholds_data = THRESHOLDS[self.measure_id]
        
        # For parametric measures with multiple parameter sets
        if self._has_parameter_sets():
            if param_set not in thresholds_data:
                raise ValueError(f"Parameter set '{param_set}' not found for measure: {self.measure_id}")
            
            # Use the specific parameter set
            thresholds_data = thresholds_data[param_set]
        
        # Now access method and categories
        if method not in thresholds_data:
            raise ValueError(f"Method '{method}' not available for measure: {self.measure_id}")
        
        if num_categories not in thresholds_data[method]:
            raise ValueError(f"No {method} thresholds for {num_categories} categories in measure: {self.measure_id}")
        
        return thresholds_data[method][num_categories]
    
    def _classify_value(self, value: float, num_categories: int, method: str = "kmeans", 
                        param_set: str = "default") -> str:
        """
        Classify a value based on thresholds.
        
        Args:
            value: The value to classify
            num_categories: Number of categories for classification
            method: Classification method (kmeans or percentile)
            param_set: Parameter set to use for threshold lookup
            
        Returns:
            Classification label
        """
        thresholds = self._get_thresholds(num_categories, method, param_set)
        labels = CATEGORY_LABELS.get(num_categories, [f"category_{i+1}" for i in range(num_categories)])
        
        # Determine category based on thresholds
        for i, threshold in enumerate(thresholds):
            if value < threshold:
                return labels[i]
        
        return labels[-1]  # If value is higher than all thresholds
    
    def _get_all_classifications(self, value: float, param_set: str = "default") -> Dict[str, Any]:
        """
        Return all available classifications for the value.
        
        Args:
            value: The value to classify
            param_set: Parameter set to use for threshold lookup
            
        Returns:
            Dictionary with classifications
        """
        result = {"value": value, "classifications": {}}
        
        # Get appropriate thresholds data
        if self.measure_id not in THRESHOLDS:
            return result
            
        thresholds_data = THRESHOLDS[self.measure_id]
        
        # For parametric measures with multiple parameter sets
        if self._has_parameter_sets():
            if param_set not in thresholds_data:
                return {
                    "value": value, 
                    "classifications": {}, 
                    "error": f"Parameter set '{param_set}' not found"
                }
            
            # Use the specific parameter set
            thresholds_data = thresholds_data[param_set]
        
        # Now collect all available classifications
        for method_name, method_data in thresholds_data.items():
            # Skip parameter definitions
            if method_name == "_params":
                continue
                
            result["classifications"][method_name] = {}
            
            for num_cats in method_data:
                try:
                    category = self._classify_value(value, num_cats, method_name, param_set)
                    result["classifications"][method_name][num_cats] = category
                except ValueError:
                    # Skip if this combination is not available
                    pass
        
        return result

    @property
    def last_result(self) -> Optional[float]:
        return self._cached_result

class ParametricPolarizationMeasure(PolarizationMeasure):
    """Base class for polarization measures with parameters."""

    def __init__(self, **parameters) -> None:
        super().__init__()
        self.parameters = parameters

    def update_parameters(self, **parameters) -> None:
        self.parameters.update(parameters)
        self._cached_result = None
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return the current parameters of the measure."""
        return self.parameters.copy()
