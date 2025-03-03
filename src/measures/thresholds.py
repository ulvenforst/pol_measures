"""
Threshold data for polarization measure classification.
These thresholds determine the boundaries for classifying polarization values.
"""

THRESHOLDS = {
    "EstebanRay": {
        # Parameter set definitions
        "_params": {
            "default": {"alpha": 0.8}
        },
        # Default parameter set thresholds (alpha=0.8)
        "default": {
            # K-means clustering thresholds
            "kmeans": {
                3: [0.3791, 0.5141],  # Low, medium, high boundaries
                4: [0.3501, 0.4467, 0.5660],  # Very low, low, high, very high
                5: [0.3196, 0.3973, 0.4809, 0.5921]  # Very low to very high
            },
            # Percentile-based thresholds
            "percentile": {
                3: [0.3609, 0.4466],  # 33rd/66th percentiles
                4: [0.3407, 0.4031, 0.4761],  # 25th/50th/75th percentiles
                5: [0.3275, 0.3781, 0.4295, 0.4965]  # 20th/40th/60th/80th percentiles
            }
        }
    },
    
    "BiPol": {
        # BiPol is non-parametric, so we don't include _params
        "kmeans": {
            3: [0.4525, 0.6267],
            4: [0.3940, 0.5317, 0.6702],
            5: [0.3543, 0.4730, 0.5816, 0.7024]
        },
        "percentile": {
            3: [0.4726, 0.5994],  # 33rd/66th percentiles
            4: [0.4380, 0.5376, 0.6383],  # 25th/50th/75th percentiles
            5: [0.4140, 0.5000, 0.5756, 0.6627]  # 20th/40th/60th/80th percentiles
        }
    },
    
    "MECNormalized": {
        # Parameter set definitions
        "_params": {
            "default": {"alpha": 2.0, "beta": 1.15}
        },
        # Default parameter set thresholds (alpha=2.0, beta=1.15)
        "default": {
            # K-means clustering thresholds
            "kmeans": {
                3: [0.2233, 0.3801],
                4: [0.1718, 0.2841, 0.4271],
                5: [0.1513, 0.2421, 0.3369, 0.4689]
            },
            # Percentile-based thresholds
            "percentile": {
                3: [0.1971, 0.2922],  # 33rd/66th percentiles
                4: [0.1747, 0.2408, 0.3274],  # 25th/50th/75th percentiles
                5: [0.1581, 0.2141, 0.2716, 0.3503]  # 20th/40th/60th/80th percentiles
            }
        }
    }
}

# Category labels for different classification schemes
CATEGORY_LABELS = {
    2: ["low", "high"],
    3: ["low", "medium", "high"],
    4: ["very_low", "low", "high", "very_high"],
    5: ["very_low", "low", "medium", "high", "very_high"]
}
