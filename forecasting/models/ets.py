"""
ETS Model Implementation for Time Series Forecasting
"""
import numpy as np
import pandas as pd
from typing import Optional
from statsmodels.tsa.exponential_smoothing.ets import ETSModel as StatsETSModel


class ETSModel:
    """
    ETS (Error, Trend, Seasonality) model for time series forecasting
    """
    
    def __init__(self, 
                 error_type: str = 'add',
                 trend_type: str = 'add',
                 seasonal_type: str = 'add',
                 seasonal_periods: Optional[int] = None):
        """
        Initialize ETS model
        
        Args:
            error_type: Type of error component ('add' or 'mul')
            trend_type: Type of trend component ('add', 'mul', or None)
            seasonal_type: Type of seasonal component ('add', 'mul', or None)
            seasonal_periods: Number of periods in a complete seasonal cycle
        """
        self.error_type = error_type
        self.trend_type = trend_type
        self.seasonal_type = seasonal_type
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.fitted_model = None
        
    def fit(self, data: np.ndarray):
        """
        Train the ETS model
        
        Args:
            data: Training time series data
        """
        # Convert to pandas Series with proper index
        ts_data = pd.Series(data, index=pd.RangeIndex(len(data)))
        
        # Create ETS model
        self.model = StatsETSModel(
            ts_data,
            error=self.error_type,
            trend=self.trend_type,
            seasonal=self.seasonal_type,
            seasonal_periods=self.seasonal_periods
        )
        
        # Fit the model
        self.fitted_model = self.model.fit(disp=False)
        
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
            
        # Generate forecasts
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