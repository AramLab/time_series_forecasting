"""
Prophet Model Implementation for Time Series Forecasting
"""
import numpy as np
import pandas as pd
from typing import Optional
from fbprophet import Prophet
from datetime import datetime, timedelta


class ProphetModel:
    """
    Prophet model for time series forecasting (Facebook's forecasting tool)
    """
    
    def __init__(self, 
                 growth: str = 'linear',
                 changepoint_prior_scale: float = 0.05,
                 seasonality_prior_scale: float = 10.0,
                 holidays_prior_scale: float = 10.0,
                 seasonality_mode: str = 'additive',
                 yearly_seasonality: str = 'auto',
                 weekly_seasonality: str = 'auto',
                 daily_seasonality: str = 'auto'):
        """
        Initialize Prophet model
        
        Args:
            growth: Growth model ('linear' or 'logistic')
            changepoint_prior_scale: Flexibility of automatic changepoint selection
            seasonality_prior_scale: Strength of seasonality model
            holidays_prior_scale: Strength of holidays model
            seasonality_mode: 'additive' or 'multiplicative'
            yearly_seasonality: Fit yearly seasonality
            weekly_seasonality: Fit weekly seasonality
            daily_seasonality: Fit daily seasonality
        """
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None
        self.fitted_model = None
        
    def prepare_data(self, data: np.ndarray) -> pd.DataFrame:
        """
        Prepare data in the format required by Prophet
        
        Args:
            data: Input time series data
            
        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        # Create date range starting from today
        dates = pd.date_range(start=datetime.today(), periods=len(data), freq='D')
        
        df = pd.DataFrame({
            'ds': dates,
            'y': data
        })
        
        return df
    
    def fit(self, data: np.ndarray):
        """
        Train the Prophet model
        
        Args:
            data: Training time series data
        """
        df = self.prepare_data(data)
        
        # Initialize Prophet model
        self.model = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality
        )
        
        # Fit the model
        self.fitted_model = self.model.fit(df)
        
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
            
        # Create future dataframe
        future_df = self.model.make_future_dataframe(periods=steps)
        
        # Make predictions
        forecast = self.model.predict(future_df)
        
        # Extract the predicted values for the future periods
        predicted_values = forecast['yhat'].iloc[-steps:].values
        
        return predicted_values
    
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