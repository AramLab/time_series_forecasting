"""
Utility functions for calculating forecasting metrics
"""
import numpy as np
from typing import Union, Tuple


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Root Mean Square Error
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        RMSE value
    """
    actual, predicted = np.array(actual), np.array(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MAE value
    """
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs(actual - predicted))


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Mean Absolute Percentage Error
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MAPE value
    """
    actual, predicted = np.array(actual), np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        SMAPE value
    """
    actual, predicted = np.array(actual), np.array(predicted)
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))


def calculate_mase(actual: np.ndarray, predicted: np.ndarray, seasonal_period: int = 1) -> float:
    """
    Calculate Mean Absolute Scaled Error
    
    Args:
        actual: Actual values
        predicted: Predicted values
        seasonal_period: Seasonal period for scaling (default=1 for non-seasonal)
        
    Returns:
        MASE value
    """
    actual, predicted = np.array(actual), np.array(predicted)
    
    # Calculate mean absolute error
    mae = np.mean(np.abs(actual - predicted))
    
    # Calculate scaling factor using naive forecast
    if len(actual) <= seasonal_period:
        # If series too short, use simple differences
        scale = np.mean(np.abs(np.diff(actual)))
    else:
        scale = np.mean(np.abs(actual[seasonal_period:] - actual[:-seasonal_period]))
    
    # Avoid division by zero
    if scale == 0:
        scale = 1e-8
        
    return mae / scale


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray, seasonal_period: int = 1) -> dict:
    """
    Calculate all forecasting metrics
    
    Args:
        actual: Actual values
        predicted: Predicted values
        seasonal_period: Seasonal period for MASE calculation
        
    Returns:
        Dictionary with all calculated metrics
    """
    actual, predicted = np.array(actual), np.array(predicted)
    
    # Ensure arrays have the same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    # Calculate all metrics
    rmse = calculate_rmse(actual, predicted)
    mae = calculate_mae(actual, predicted)
    mape = calculate_mape(actual, predicted)
    smape = calculate_smape(actual, predicted)
    mase = calculate_mase(actual, predicted, seasonal_period)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'smape': smape,
        'mase': mase
    }