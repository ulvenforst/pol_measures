"""Optimization helpers for polarization measures."""

from __future__ import annotations

import math

import numpy as np


def mec_bisection_value(
    x: np.ndarray,
    weights: np.ndarray,
    *,
    alpha: float,
    beta: float,
    epsilon: float = 1e-8,
) -> float:
    """
    Compute MEC by bisection on the first-order condition for beta > 1.

    For
        F(y) = sum_i weights_i^alpha * |x_i - y|^beta,
    the derivative satisfies
        F'(y) / beta = sum_i weights_i^alpha * sign(y - x_i) * |y - x_i|^(beta - 1).

    When alpha > 0 and beta > 1, this derivative-root function is monotone
    increasing, so the minimizer can be found by bisection. The returned value is
    F(y*) evaluated at the midpoint of the final bracket.

    This helper assumes the caller has already applied the public histogram
    validation/normalization conventions. It is intended as a reference optimizer
    for benchmarks and tests; the public MEC class still uses SciPy's bounded
    scalar optimizer.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if beta <= 1:
        raise ValueError("bisection MEC requires beta > 1")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    x = np.asarray(x, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    if x.ndim != 1 or weights.ndim != 1:
        raise ValueError("x and weights must be one-dimensional")
    if x.shape != weights.shape:
        raise ValueError("x and weights must have the same shape")
    if x.size < 2:
        raise ValueError("At least two points are required")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x values must be strictly increasing")
    if np.any(weights < 0):
        raise ValueError("All weights must be non-negative")
    if not np.any(weights > 0):
        raise ValueError("At least one weight must be positive")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(weights)):
        raise ValueError("x and weights must be finite")

    lo = float(x[0])
    hi = float(x[-1])
    width = hi - lo
    if width <= epsilon:
        y_star = 0.5 * (lo + hi)
        return float(np.sum((weights**alpha) * (np.abs(x - y_star) ** beta)))

    weights_alpha = weights**alpha
    iterations = max(1, math.ceil(math.log2(width / epsilon)))

    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        delta = mid - x
        derivative_without_beta = float(
            np.sum(weights_alpha * np.sign(delta) * (np.abs(delta) ** (beta - 1.0)))
        )
        if derivative_without_beta < 0.0:
            lo = mid
        else:
            hi = mid

    y_star = 0.5 * (lo + hi)
    return float(np.sum(weights_alpha * (np.abs(x - y_star) ** beta)))
