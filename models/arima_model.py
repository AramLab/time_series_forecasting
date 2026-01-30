import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')


def check_stationarity(timeseries, max_lags=100):
    """Check if time series is stationary using Augmented Dickey-Fuller test"""
    # Reduce lags if series is too short
    n_samples = len(timeseries)
    effective_max_lags = min(max_lags, n_samples // 4)  # Rule of thumb: max 25% of data
    
    if effective_max_lags < 1:
        return False  # Too short to test properly
    
    try:
        result = adfuller(timeseries, maxlag=effective_max_lags)
        p_value = result[1]
        return p_value <= 0.05
    except:
        return False  # If test fails, assume non-stationary


class ARIMAModel:
    def __init__(self, max_p=5, max_q=5, max_d=2):
        """
        Initialize ARIMA model with configuration parameters
        """
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.model = None
        self.fitted_model = None
        self.order = None
    
    def fit(self, data):
        """
        Fit ARIMA model on the provided data
        """
        # Determine differencing order (d)
        d = 0
        if not check_stationarity(data):
            # Try differencing
            temp_diff_1 = data.diff().dropna()
            if check_stationarity(temp_diff_1):
                d = 1
            else:
                temp_diff_2 = temp_diff_1.diff().dropna()
                if check_stationarity(temp_diff_2) and len(temp_diff_2) > 10:
                    d = 2
                else:
                    d = 1  # Default to first differencing if still not stationary
        
        # Find best (p,q) orders
        best_aic = np.inf
        best_order = None
        best_fitted_model = None
        
        # Limit search space for smaller datasets
        max_p_actual = min(self.max_p, max(1, len(data)//4))  # Prevent overparameterization
        max_q_actual = min(self.max_q, max(1, len(data)//4))
        
        for p in range(0, max_p_actual + 1):
            for q in range(0, max_q_actual + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                        best_fitted_model = fitted_model
                except:
                    continue
        
        if best_fitted_model is None:
            # If no model fits, try with minimal parameters
            d = min(d, 1)  # Limit differencing
            for p in range(0, min(2, max_p_actual + 1)):
                for q in range(0, min(2, max_q_actual + 1)):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            best_fitted_model = fitted_model
                        break
                    except:
                        continue
                if best_fitted_model is not None:
                    break
        
        if best_fitted_model is None:
            # If still no model found, use basic parameters
            model = ARIMA(data, order=(1, 1, 1))
            best_fitted_model = model.fit()
            best_order = (1, 1, 1)
        
        self.fitted_model = best_fitted_model
        self.order = best_order
        return self

    def predict(self, steps):
        """
        Forecast future values using the trained ARIMA model
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Generate forecast
        forecast_result = self.fitted_model.forecast(steps=steps)
        forecast_values = forecast_result.values
        
        return forecast_values