"""
Configuration settings for the time series forecasting project
"""
import os
from typing import Dict, Any


class Config:
    """
    Configuration class for time series forecasting project
    """
    
    def __init__(self):
        # Model default parameters
        self.lstm_defaults = {
            'sequence_length': 10,
            'lstm_units': 50,
            'dropout_rate': 0.2,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001
        }
        
        self.arima_defaults = {
            'order': None,  # Will be auto-detected
            'seasonal_order': None,
            'maxiter': 50
        }
        
        self.ets_defaults = {
            'error_type': 'add',
            'trend_type': 'add',
            'seasonal_type': 'add',
            'seasonal_periods': None
        }
        
        self.prophet_defaults = {
            'growth': 'linear',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'seasonality_mode': 'additive',
            'yearly_seasonality': 'auto',
            'weekly_seasonality': 'auto',
            'daily_seasonality': 'auto'
        }
        
        self.ceemdan_defaults = {
            'ensemble_size': 10,
            'max_imfs': 10
        }
        
        # Data processing defaults
        self.data_defaults = {
            'test_size': 12,
            'max_series': 10,
            'seasonal_period': 12
        }
        
        # Parallel processing settings
        self.parallel_defaults = {
            'use_multiprocessing': True,
            'num_processes': os.cpu_count(),
            'chunk_size': 1
        }
        
        # Results and output settings
        self.output_defaults = {
            'results_dir': './results',
            'plots_dir': './results/plots',
            'csv_dir': './results/csv',
            'save_plots': True,
            'save_csv': True
        }
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get default configuration for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with default parameters
        """
        configs = {
            'lstm': self.lstm_defaults,
            'arima': self.arima_defaults,
            'ets': self.ets_defaults,
            'prophet': self.prophet_defaults,
            'ceemdan_lstm': {**self.ceemdan_defaults, **self.lstm_defaults},
            'ceemdan_arima': {**self.ceemdan_defaults, **self.arima_defaults},
            'ceemdan_ets': {**self.ceemdan_defaults, **self.ets_defaults},
            'ceemdan_prophet': {**self.ceemdan_defaults, **self.prophet_defaults}
        }
        
        if model_name.lower() in configs:
            return configs[model_name.lower()]
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get default data processing configuration
        
        Returns:
            Dictionary with data processing parameters
        """
        return self.data_defaults
    
    def get_parallel_config(self) -> Dict[str, Any]:
        """
        Get default parallel processing configuration
        
        Returns:
            Dictionary with parallel processing parameters
        """
        return self.parallel_defaults
    
    def get_output_config(self) -> Dict[str, Any]:
        """
        Get default output configuration
        
        Returns:
            Dictionary with output parameters
        """
        return self.output_defaults