"""
Data Loader for Time Series Datasets
Supports M3, M4, and other popular time series datasets
"""
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class DataLoader:
    """
    Data loader for various time series datasets including M3, M4, and custom formats
    """
    
    def __init__(self):
        self.datasets = {
            'm3': self.load_m3_data,
            'm4': self.load_m4_data,
            'custom': self.load_custom_data
        }
        
    def load_m3_data(self, limit: Optional[int] = None) -> List[np.ndarray]:
        """
        Load M3 competition data
        
        Args:
            limit: Maximum number of series to load
            
        Returns:
            List of time series data arrays
        """
        # In a real implementation, this would load from actual M3 files
        # For now, we'll generate some sample data similar to M3
        print("Loading M3-like synthetic data...")
        
        series_list = []
        num_series = min(limit, 10) if limit else 10  # Using 10 as default for demo
        
        for i in range(num_series):
            # Generate series of different lengths to simulate M3 variety
            length = np.random.randint(24, 200)  # M3 series range from 14 to 1476
            trend_factor = np.random.uniform(0.01, 0.05)
            seasonality_freq = np.random.choice([4, 12])  # Quarterly or monthly
            
            t = np.arange(length)
            trend = trend_factor * t
            seasonal = 0.5 * np.sin(2 * np.pi * t / seasonality_freq)
            noise = np.random.normal(0, 0.1, length)
            
            series = 10 + trend + seasonal + noise
            series_list.append(series)
            
        return series_list
    
    def load_m4_data(self, limit: Optional[int] = None) -> List[np.ndarray]:
        """
        Load M4 competition data
        
        Args:
            limit: Maximum number of series to load
            
        Returns:
            List of time series data arrays
        """
        # In a real implementation, this would load from actual M4 files
        # For now, we'll generate some sample data similar to M4
        print("Loading M4-like synthetic data...")
        
        series_list = []
        num_series = min(limit, 10) if limit else 10  # Using 10 as default for demo
        
        for i in range(num_series):
            # Generate series of different frequencies to simulate M4 variety
            freq_types = ['hourly', 'daily', 'weekly', 'monthly', 'quarterly', 'yearly']
            freq_map = {'hourly': 24, 'daily': 7, 'weekly': 52, 'monthly': 12, 'quarterly': 4, 'yearly': 1}
            
            # Randomly select frequency type
            freq_type = np.random.choice(freq_types)
            seasonality_period = freq_map[freq_type]
            
            # Generate series length based on frequency
            if freq_type in ['hourly']:
                length = np.random.randint(1000, 20000)
            elif freq_type in ['daily']:
                length = np.random.randint(100, 1000)
            elif freq_type in ['weekly']:
                length = np.random.randint(50, 1000)
            elif freq_type in ['monthly']:
                length = np.random.randint(24, 200)
            elif freq_type in ['quarterly']:
                length = np.random.randint(12, 100)
            else:  # yearly
                length = np.random.randint(10, 50)
                
            t = np.arange(length)
            trend_factor = np.random.uniform(0.005, 0.03)
            trend = trend_factor * t
            seasonal = 0.3 * np.sin(2 * np.pi * t / seasonality_period)
            noise = np.random.normal(0, 0.05, length)
            
            series = 5 + trend + seasonal + noise
            series_list.append(series)
            
        return series_list
    
    def load_custom_data(self, filepath: str) -> List[np.ndarray]:
        """
        Load custom time series data from file
        
        Args:
            filepath: Path to the data file
            
        Returns:
            List of time series data arrays
        """
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        # Try to load as CSV
        try:
            df = pd.read_csv(filepath)
            
            # If there are multiple columns, treat each as a separate series
            if df.shape[1] > 1:
                series_list = []
                for col in df.columns:
                    series = df[col].dropna().values
                    if len(series) > 0:
                        series_list.append(series)
            else:
                # Single column - treat as one series
                series = df.iloc[:, 0].dropna().values
                series_list = [series] if len(series) > 0 else []
                
            return series_list
        except Exception as e:
            raise ValueError(f"Could not load data from {filepath}: {str(e)}")
    
    def load_dataset(self, dataset_name: str, limit: Optional[int] = None, **kwargs) -> List[np.ndarray]:
        """
        Load a specific dataset
        
        Args:
            dataset_name: Name of the dataset ('m3', 'm4', 'custom')
            limit: Maximum number of series to load
            **kwargs: Additional arguments for specific loaders
            
        Returns:
            List of time series data arrays
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not supported. Supported: {list(self.datasets.keys())}")
            
        if dataset_name == 'custom':
            filepath = kwargs.get('filepath')
            if not filepath:
                raise ValueError("For 'custom' dataset, filepath must be provided")
            return self.load_custom_data(filepath)
        else:
            return self.datasets[dataset_name](limit)