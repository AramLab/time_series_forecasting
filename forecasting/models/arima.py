"""
ARIMA Model Implementation for Time Series Forecasting
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from statsmodels.tsa.arima.model import ARIMA as SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")


class ARIMAModel:
    """
    ARIMA (AutoRegressive Integrated Moving Average) model for time series forecasting
    """
    
    def __init__(self, 
                 order: Optional[Tuple[int, int, int]] = None,
                 seasonal_order: Optional[Tuple[int, int, int, int]] = None,
                 maxiter: int = 50):
        """
        Initialize ARIMA model
        
        Args:
            order: (p, d, q) order of the ARIMA model
            seasonal_order: (P, D, Q, S) seasonal order of the model
            maxiter: Maximum number of iterations for fitting
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.maxiter = maxiter
        self.model = None
        self.fitted_model = None
        
    def determine_order(self, data: np.ndarray) -> Tuple[int, int, int]:
        """
        Automatically determine optimal ARIMA order using AIC criterion
        
        Args:
            data: Input time series data
            
        Returns:
            Optimal (p, d, q) order
        """
        # Determine differencing order (d) using ADF test
        d = 0
        adf_result = adfuller(data)
        if adf_result[1] > 0.05:  # p-value > 0.05, not stationary
            d = 1
            data_diff = np.diff(data)
            adf_result = adfuller(data_diff)
            if adf_result[1] > 0.05:
                d = 2
                data_diff = np.diff(data, n=2)
                
        # Find optimal (p, q) using grid search
        best_aic = float('inf')
        best_order = (0, d, 0)
        
        for p in range(0, 4):
            for q in range(0, 4):
                try:
                    model = SARIMAX(data, order=(p, d, q))
                    fitted_model = model.fit(disp=False, maxiter=self.maxiter)
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                except:
                    continue
                    
        return best_order
    
    def fit(self, data: np.ndarray):
        """
        Train the ARIMA model
        
        Args:
            data: Training time series data
        """
        if self.order is None:
            self.order = self.determine_order(data)
            
        self.model = SARIMAX(data, order=self.order)
        self.fitted_model = self.model.fit(disp=False, maxiter=self.maxiter)
        
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            steps: Number of steps to predict
            
        Returns:
            Predicted values
        """
        if self.fitted_model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        forecast = self.fitted_model.forecast(steps=steps)
        return forecast.values
    
    def fit_predict(self, data: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Fit the model and make predictions
        
        Args:
            data: Training time series data
            steps: Number of steps to predict
            
        Returns:
            Predicted values
        """
        self.fit(data)
        return self.predict(steps)