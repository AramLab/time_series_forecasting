"""
Optimization Package

Contains modules for optimizing time series forecasting models:
- ceemdan_optimizer: CEEMDAN parameter optimization
- complexity_reduction: Computational complexity reduction techniques
"""

from .ceemdan_optimizer import (
    CEEMDANOptimizer,
    optimize_ceemdan_params
)

from .complexity_reduction import (
    ComplexityReducer,
    AdaptiveHybridForecaster,
    estimate_forecasting_complexity
)

__all__ = [
    'CEEMDANOptimizer',
    'optimize_ceemdan_params',
    'ComplexityReducer',
    'AdaptiveHybridForecaster',
    'estimate_forecasting_complexity'
]

__version__ = '1.0.0'
