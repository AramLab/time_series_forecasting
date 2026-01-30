from statsforecast import StatsForecast
from statsforecast.models import AutoETS
import pandas as pd
import numpy as np


class ETSModel:
    def __init__(self, season_length=None, model='ZZZ'):
        """
        Initialize ETS model with configuration parameters
        """
        self.season_length = season_length
        self.model = model
        self.sf = None
        self.freq = None
    
    def fit(self, data, freq=None):
        """
        Fit ETS model on the provided data
        """
        # Convert data to required format for statsforecast
        if isinstance(data, pd.Series):
            ds_values = data.index
            y_values = data.values
            self.freq = data.index.freqstr if hasattr(data.index, 'freqstr') else freq
        else:
            # If data is just an array, we'll need to create a date index
            ds_values = pd.date_range(start='2000-01-01', periods=len(data), freq=freq or 'D')
            y_values = data
            self.freq = freq or 'D'

        df = pd.DataFrame({
            'unique_id': 1,
            'ds': ds_values,
            'y': y_values
        })

        # Set season_length based on data if not specified
        if self.season_length is None:
            # Simple heuristic: if we have more than 2 years of monthly data, use season_length=12
            if len(data) > 24 and freq in ['M', 'MS']:
                self.season_length = 12
            elif len(data) > 8 and freq in ['W', 'W-SUN']:
                self.season_length = 4
            elif len(data) > 24 and freq in ['D']:
                self.season_length = 7
            else:
                self.season_length = 1

        # Create and fit the model
        models = [AutoETS(season_length=self.season_length, model=self.model)]
        self.sf = StatsForecast(models=models, freq=self.freq, n_jobs=1)
        self.sf.fit(df)
        return self

    def predict(self, steps):
        """
        Forecast future values using the trained ETS model
        """
        if self.sf is None:
            raise ValueError("Model must be fitted before making predictions")
        
        # Generate forecast
        forecast_df = self.sf.predict(h=steps)
        forecast_values = forecast_df[f'AutoETS'].values
        
        return forecast_values