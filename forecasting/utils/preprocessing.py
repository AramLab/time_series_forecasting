"""
Utility functions for preprocessing time series data
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def remove_outliers_iqr(data: np.ndarray, multiplier: float = 1.5) -> np.ndarray:
    """
    Remove outliers using Interquartile Range (IQR) method
    
    Args:
        data: Input time series data
        multiplier: Multiplier for IQR (default 1.5)
        
    Returns:
        Data with outliers removed/replaced
    """
    data = np.array(data)
    
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    # Replace outliers with median
    cleaned_data = np.where((data < lower_bound) | (data > upper_bound), 
                            np.median(data), data)
    
    return cleaned_data


def remove_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Remove outliers using Z-score method
    
    Args:
        data: Input time series data
        threshold: Z-score threshold (default 3.0)
        
    Returns:
        Data with outliers removed/replaced
    """
    data = np.array(data)
    
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    cleaned_data = np.where(z_scores > threshold, np.median(data), data)
    
    return cleaned_data


def detrend_series(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove linear trend from time series
    
    Args:
        data: Input time series data
        
    Returns:
        Detrended data and trend component
    """
    t = np.arange(len(data))
    
    # Fit linear trend
    coeffs = np.polyfit(t, data, 1)
    trend = np.polyval(coeffs, t)
    
    # Remove trend
    detrended = data - trend
    
    return detrended, trend


def deseasonalize_series(data: np.ndarray, period: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove seasonal component from time series using moving average
    
    Args:
        data: Input time series data
        period: Seasonal period
        
    Returns:
        Deseasonalized data and seasonal component
    """
    # Calculate seasonal averages
    seasonal_avg = np.zeros(period)
    for i in range(period):
        seasonal_avg[i] = np.mean(data[i::period])
    
    # Create full seasonal pattern
    seasonal_pattern = np.tile(seasonal_avg, int(len(data)/period) + 1)[:len(data)]
    
    # Remove seasonal component
    deseasonalized = data - seasonal_pattern
    
    return deseasonalized, seasonal_pattern


def normalize_series(data: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, dict]:
    """
    Normalize time series data
    
    Args:
        data: Input time series data
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized data and normalization parameters
    """
    data = np.array(data)
    
    if method == 'minmax':
        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val)
        params = {'min': min_val, 'max': max_val}
        
    elif method == 'zscore':
        mean_val = np.mean(data)
        std_val = np.std(data)
        normalized = (data - mean_val) / std_val
        params = {'mean': mean_val, 'std': std_val}
        
    elif method == 'robust':
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))
        normalized = (data - median_val) / mad
        params = {'median': median_val, 'mad': mad}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_series(normalized_data: np.ndarray, params: dict, method: str = 'minmax') -> np.ndarray:
    """
    Denormalize time series data using stored parameters
    
    Args:
        normalized_data: Normalized time series data
        params: Normalization parameters
        method: Normalization method used
        
    Returns:
        Denormalized data
    """
    if method == 'minmax':
        return normalized_data * (params['max'] - params['min']) + params['min']
    elif method == 'zscore':
        return normalized_data * params['std'] + params['mean']
    elif method == 'robust':
        return normalized_data * params['mad'] + params['median']
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def preprocess_data(data: np.ndarray, 
                   remove_outliers: bool = True,
                   outlier_method: str = 'iqr',
                   detrend: bool = False,
                   deseasonalize: bool = False,
                   seasonal_period: int = 12,
                   normalize: bool = True,
                   norm_method: str = 'minmax') -> Tuple[np.ndarray, dict]:
    """
    Complete preprocessing pipeline for time series data
    
    Args:
        data: Input time series data
        remove_outliers: Whether to remove outliers
        outlier_method: Method for outlier removal ('iqr', 'zscore')
        detrend: Whether to remove trend
        deseasonalize: Whether to remove seasonal component
        seasonal_period: Seasonal period for deseasonalization
        normalize: Whether to normalize data
        norm_method: Normalization method
        
    Returns:
        Preprocessed data and preprocessing parameters
    """
    original_data = np.array(data)
    processed_data = original_data.copy()
    params = {}
    
    # Remove outliers
    if remove_outliers:
        if outlier_method == 'iqr':
            processed_data = remove_outliers_iqr(processed_data)
        elif outlier_method == 'zscore':
            processed_data = remove_outliers_zscore(processed_data)
        else:
            raise ValueError(f"Unknown outlier method: {outlier_method}")
    
    # Detrend
    if detrend:
        processed_data, trend = detrend_series(processed_data)
        params['trend'] = trend
    
    # Deseasonalize
    if deseasonalize:
        processed_data, seasonal = deseasonalize_series(processed_data, seasonal_period)
        params['seasonal'] = seasonal
    
    # Normalize
    if normalize:
        processed_data, norm_params = normalize_series(processed_data, norm_method)
        params['normalization'] = norm_params
        params['norm_method'] = norm_method
    
    params['original_shape'] = original_data.shape
    
    return processed_data, params