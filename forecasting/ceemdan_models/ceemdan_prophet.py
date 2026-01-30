"""
CEEMDAN+Prophet Model Implementation for Time Series Forecasting
"""
import numpy as np
import pandas as pd
from typing import List
from PyEMD import CEEMDAN
from ..models.prophet import ProphetModel


class CEEMDANProphet:
    """
    CEEMDAN (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise) + Prophet model
    for time series forecasting
    """
    
    def __init__(self, 
                 ensemble_size: int = 10,
                 max_imfs: int = 10,
                 **prophet_kwargs):
        """
        Initialize CEEMDAN+Prophet model
        
        Args:
            ensemble_size: Number of ensemble members for CEEMDAN
            max_imfs: Maximum number of IMFs to extract
            **prophet_kwargs: Arguments to pass to Prophet model
        """
        self.ensemble_size = ensemble_size
        self.max_imfs = max_imfs
        self.prophet_kwargs = prophet_kwargs
        self.ceemdan = CEEMDAN(ensemble_size=ensemble_size, max_imfs=max_imfs)
        self.prophet_models = {}
        self.is_fitted = False
        
    def decompose(self, data: np.ndarray) -> List[np.ndarray]:
        """
        Decompose time series using CEEMDAN
        
        Args:
            data: Input time series data
            
        Returns:
            List of IMFs (Intrinsic Mode Functions) and residual
        """
        imfs = self.ceemdan(data)
        return imfs
    
    def fit(self, data: np.ndarray):
        """
        Train the CEEMDAN+Prophet model
        
        Args:
            data: Training time series data
        """
        # Decompose the time series
        imfs = self.decompose(data)
        
        # Train a Prophet model for each IMF
        self.prophet_models = {}
        for i, imf in enumerate(imfs):
            prophet_model = ProphetModel(**self.prophet_kwargs)
            prophet_model.fit(imf)
            self.prophet_models[f'imf_{i}'] = prophet_model
            
        self.is_fitted = True
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Get predictions from each IMF model
        predictions = []
        for key, model in self.prophet_models.items():
            pred = model.predict(steps)
            predictions.append(pred)
        
        # Sum predictions from all IMFs
        final_prediction = np.sum(predictions, axis=0)
        
        return final_prediction
    
    def fit_predict(self, data: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Fit the model and make predictions
        
        Args:
            data: Training time series data
            steps: Number of steps to predict
            
        Returns:
            Predicted values
        """
        # Decompose the time series
        imfs = self.decompose(data)
        
        # Train a Prophet model for each IMF and make predictions
        predictions = []
        for i, imf in enumerate(imfs):
            prophet_model = ProphetModel(**self.prophet_kwargs)
            pred = prophet_model.fit_predict(imf, steps)
            predictions.append(pred)
        
        # Sum predictions from all IMFs
        final_prediction = np.sum(predictions, axis=0)
        
        return final_prediction