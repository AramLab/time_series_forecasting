from prophet import Prophet
import pandas as pd
import numpy as np


class ProphetModel:
    def __init__(self, growth='linear', changepoint_prior_scale=0.05, seasonality_prior_scale=10.0,
                 seasonality_mode='additive', yearly_seasonality=True, weekly_seasonality=False,
                 daily_seasonality=False, changepoint_range=0.8, interval_width=0.95):
        """
        Initialize Prophet model with configuration parameters
        """
        self.growth = growth
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_range = changepoint_range
        self.interval_width = interval_width
        self.model = None
        self.freq = None
    
    def fit(self, data, freq='D'):
        """
        Fit Prophet model on the provided data
        """
        # Convert data to required format for Prophet
        if isinstance(data, pd.Series):
            df = pd.DataFrame({'ds': data.index, 'y': data.values})
            self.freq = data.index.freqstr if hasattr(data.index, 'freqstr') else freq
        else:
            # If data is just an array, we'll need to create a date index
            dates = pd.date_range(start='2000-01-01', periods=len(data), freq=freq)
            df = pd.DataFrame({'ds': dates, 'y': data})
            self.freq = freq

        # Create and fit the model
        self.model = Prophet(
            growth=self.growth,
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_range=self.changepoint_range,
            interval_width=self.interval_width
        )

        self.model.fit(df)
        return self

    def predict(self, steps):
        """
        Forecast future values using the trained Prophet model
        """
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create future dataframe
        future_df = self.model.make_future_dataframe(periods=steps, freq=self.freq)
        future_df = future_df.tail(steps)  # Get only the future periods we need
        
        # Generate forecast
        forecast = self.model.predict(future_df)
        forecast_values = forecast['yhat'].values
        
        return forecast_values