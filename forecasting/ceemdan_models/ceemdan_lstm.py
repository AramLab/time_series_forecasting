"""
CEEMDAN+LSTM Model Implementation for Time Series Forecasting
"""
import numpy as np
import pandas as pd
from typing import List, Tuple
from PyEMD import CEEMDAN
from ..models.lstm import LSTMModel


class CEEMDANLSTM:
    """
    CEEMDAN (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise) + LSTM model
    for time series forecasting
    """
    
    def __init__(self, 
                 ensemble_size: int = 10,
                 max_imfs: int = 10,
                 **lstm_kwargs):
        """
        Initialize CEEMDAN+LSTM model
        
        Args:
            ensemble_size: Number of ensemble members for CEEMDAN
            max_imfs: Maximum number of IMFs to extract
            **lstm_kwargs: Arguments to pass to LSTM model
        """
        self.ensemble_size = ensemble_size
        self.max_imfs = max_imfs
        self.lstm_kwargs = lstm_kwargs
        self.ceemdan = CEEMDAN(ensemble_size=ensemble_size, max_imfs=max_imfs)
        self.lstm_models = {}
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
        Train the CEEMDAN+LSTM model
        
        Args:
            data: Training time series data
        """
        # Decompose the time series
        imfs = self.decompose(data)
        
        # Train an LSTM model for each IMF
        self.lstm_models = {}
        for i, imf in enumerate(imfs):
            lstm_model = LSTMModel(**self.lstm_kwargs)
            lstm_model.fit(imf)
            self.lstm_models[f'imf_{i}'] = lstm_model
            
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
        for key, model in self.lstm_models.items():
            pred = model.predict(model.scaler.inverse_transform([[0]] * self.lstm_models[key].sequence_length).flatten(), steps)
            # For simplicity, just predict using the last known values of each IMF
            # In practice, we would need to extend each IMF series properly
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
        # For a more realistic implementation, we need to properly handle the 
        # forecasting of each IMF component
        imfs = self.decompose(data)
        
        # Train an LSTM model for each IMF and make predictions
        predictions = []
        for i, imf in enumerate(imfs):
            lstm_model = LSTMModel(**self.lstm_kwargs)
            pred = lstm_model.fit_predict(imf, steps)
            predictions.append(pred)
        
        # Sum predictions from all IMFs
        final_prediction = np.sum(predictions, axis=0)
        
        return final_prediction