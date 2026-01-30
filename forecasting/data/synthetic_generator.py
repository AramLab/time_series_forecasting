"""
Synthetic Time Series Generator
Generates various types of synthetic time series data for testing and experimentation
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


class SyntheticGenerator:
    """
    Generator for synthetic time series data with various patterns and characteristics
    """
    
    def __init__(self):
        pass
    
    def generate_trend_series(self, 
                              length: int = 100, 
                              trend_slope: float = 0.1,
                              noise_level: float = 0.1) -> np.ndarray:
        """
        Generate a time series with linear trend
        
        Args:
            length: Length of the series
            trend_slope: Slope of the linear trend
            noise_level: Level of random noise
            
        Returns:
            Generated time series
        """
        t = np.arange(length)
        trend = trend_slope * t
        noise = np.random.normal(0, noise_level, length)
        
        return trend + noise
    
    def generate_seasonal_series(self, 
                                 length: int = 100, 
                                 seasonal_period: int = 12,
                                 amplitude: float = 1.0,
                                 noise_level: float = 0.1) -> np.ndarray:
        """
        Generate a time series with seasonal pattern
        
        Args:
            length: Length of the series
            seasonal_period: Period of the seasonal pattern
            amplitude: Amplitude of the seasonal pattern
            noise_level: Level of random noise
            
        Returns:
            Generated time series
        """
        t = np.arange(length)
        seasonal = amplitude * np.sin(2 * np.pi * t / seasonal_period)
        noise = np.random.normal(0, noise_level, length)
        
        return seasonal + noise
    
    def generate_trend_seasonal_series(self, 
                                       length: int = 100,
                                       trend_slope: float = 0.05,
                                       seasonal_period: int = 12,
                                       seasonal_amplitude: float = 1.0,
                                       noise_level: float = 0.1) -> np.ndarray:
        """
        Generate a time series with both trend and seasonal patterns
        
        Args:
            length: Length of the series
            trend_slope: Slope of the linear trend
            seasonal_period: Period of the seasonal pattern
            seasonal_amplitude: Amplitude of the seasonal pattern
            noise_level: Level of random noise
            
        Returns:
            Generated time series
        """
        t = np.arange(length)
        trend = trend_slope * t
        seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
        noise = np.random.normal(0, noise_level, length)
        
        return trend + seasonal + noise
    
    def generate_random_walk(self, 
                             length: int = 100, 
                             drift: float = 0.0,
                             volatility: float = 1.0) -> np.ndarray:
        """
        Generate a random walk time series
        
        Args:
            length: Length of the series
            drift: Drift parameter
            volatility: Volatility parameter
            
        Returns:
            Generated time series
        """
        increments = np.random.normal(drift, volatility, length)
        series = np.cumsum(increments)
        
        return series
    
    def generate_ar_series(self, 
                           length: int = 100,
                           coefficients: List[float] = [0.5],
                           noise_level: float = 1.0) -> np.ndarray:
        """
        Generate an autoregressive (AR) time series
        
        Args:
            length: Length of the series
            coefficients: AR coefficients
            noise_level: Level of random noise
            
        Returns:
            Generated time series
        """
        series = np.zeros(length)
        p = len(coefficients)
        
        # Initialize first p values randomly
        series[:p] = np.random.normal(0, noise_level, p)
        
        # Generate the rest using AR equation
        for i in range(p, length):
            series[i] = sum(coefficients[j] * series[i-j-1] for j in range(p))
            series[i] += np.random.normal(0, noise_level)
            
        return series
    
    def generate_multiple_series(self, 
                                 num_series: int = 5,
                                 series_params: Optional[List[dict]] = None) -> List[np.ndarray]:
        """
        Generate multiple synthetic time series with different characteristics
        
        Args:
            num_series: Number of series to generate
            series_params: List of parameters for each series (if None, random parameters)
            
        Returns:
            List of generated time series
        """
        series_list = []
        
        if series_params is None:
            # Generate random parameters for each series
            for i in range(num_series):
                series_type = np.random.choice([
                    'trend', 'seasonal', 'trend_seasonal', 'random_walk', 'ar'
                ])
                
                if series_type == 'trend':
                    series = self.generate_trend_series(
                        length=np.random.randint(50, 200),
                        trend_slope=np.random.uniform(0.01, 0.1),
                        noise_level=np.random.uniform(0.05, 0.2)
                    )
                elif series_type == 'seasonal':
                    series = self.generate_seasonal_series(
                        length=np.random.randint(50, 200),
                        seasonal_period=np.random.choice([4, 12, 52]),
                        amplitude=np.random.uniform(0.5, 2.0),
                        noise_level=np.random.uniform(0.05, 0.2)
                    )
                elif series_type == 'trend_seasonal':
                    series = self.generate_trend_seasonal_series(
                        length=np.random.randint(50, 200),
                        trend_slope=np.random.uniform(0.01, 0.05),
                        seasonal_period=np.random.choice([4, 12, 52]),
                        seasonal_amplitude=np.random.uniform(0.5, 1.5),
                        noise_level=np.random.uniform(0.05, 0.15)
                    )
                elif series_type == 'random_walk':
                    series = self.generate_random_walk(
                        length=np.random.randint(50, 200),
                        drift=np.random.uniform(-0.05, 0.05),
                        volatility=np.random.uniform(0.5, 1.5)
                    )
                else:  # AR
                    series = self.generate_ar_series(
                        length=np.random.randint(50, 200),
                        coefficients=[np.random.uniform(0.1, 0.9)],
                        noise_level=np.random.uniform(0.5, 1.0)
                    )
                    
                series_list.append(series)
        else:
            # Use provided parameters
            for params in series_params:
                series_type = params.get('type', 'trend_seasonal')
                
                if series_type == 'trend':
                    series = self.generate_trend_series(**{k: v for k, v in params.items() if k != 'type'})
                elif series_type == 'seasonal':
                    series = self.generate_seasonal_series(**{k: v for k, v in params.items() if k != 'type'})
                elif series_type == 'trend_seasonal':
                    series = self.generate_trend_seasonal_series(**{k: v for k, v in params.items() if k != 'type'})
                elif series_type == 'random_walk':
                    series = self.generate_random_walk(**{k: v for k, v in params.items() if k != 'type'})
                elif series_type == 'ar':
                    series = self.generate_ar_series(**{k: v for k, v in params.items() if k != 'type'})
                else:
                    raise ValueError(f"Unknown series type: {series_type}")
                    
                series_list.append(series)
                
        return series_list
    
    def generate_with_seasonality_options(self, 
                                          length: int = 100,
                                          trend: bool = True,
                                          seasonal: bool = True,
                                          seasonal_periods: Optional[List[int]] = None,
                                          noise_level: float = 0.1) -> np.ndarray:
        """
        Generate a time series with configurable seasonality options
        
        Args:
            length: Length of the series
            trend: Whether to include a trend component
            seasonal: Whether to include a seasonal component
            seasonal_periods: List of seasonal periods to include (e.g., [12, 4] for annual and quarterly)
            noise_level: Level of random noise
            
        Returns:
            Generated time series
        """
        t = np.arange(length)
        series = np.zeros(length)
        
        if trend:
            # Add linear trend
            slope = np.random.uniform(0.01, 0.05)
            series += slope * t
            
        if seasonal and seasonal_periods:
            # Add multiple seasonal components
            for period in seasonal_periods:
                amplitude = np.random.uniform(0.5, 1.5)
                series += amplitude * np.sin(2 * np.pi * t / period)
        
        # Add noise
        noise = np.random.normal(0, noise_level, length)
        series += noise
        
        return series