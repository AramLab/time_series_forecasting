"""
LSTM Model Implementation for Time Series Forecasting
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class LSTMModel:
    """
    LSTM (Long Short-Term Memory) model for time series forecasting
    """
    
    def __init__(self, 
                 sequence_length: int = 10,
                 lstm_units: int = 50,
                 dropout_rate: float = 0.2,
                 epochs: int = 100,
                 batch_size: int = 32,
                 learning_rate: float = 0.001):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Length of input sequences
            lstm_units: Number of LSTM units
            dropout_rate: Dropout rate for regularization
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def prepare_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM training
        
        Args:
            data: Input time series data
            
        Returns:
            X, y: Feature and target arrays
        """
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i, 0])
            y.append(scaled_data[i, 0])
            
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM
        
        return X, y
    
    def build_model(self, input_shape: Tuple[int, int]):
        """
        Build LSTM neural network
        
        Args:
            input_shape: Shape of input data (timesteps, features)
        """
        self.model = Sequential([
            LSTM(units=self.lstm_units, 
                 return_sequences=True, 
                 input_shape=input_shape),
            Dropout(self.dropout_rate),
            LSTM(units=self.lstm_units, 
                 return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(units=25),
            Dense(units=1)
        ])
        
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    def fit(self, data: np.ndarray):
        """
        Train the LSTM model
        
        Args:
            data: Training time series data
        """
        X, y = self.prepare_data(data)
        
        if self.model is None:
            self.build_model((X.shape[1], X.shape[2]))
            
        self.model.fit(X, y, 
                       epochs=self.epochs, 
                       batch_size=self.batch_size, 
                       verbose=0)
        
    def predict(self, data: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            data: Input data for prediction
            steps: Number of steps to predict
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Scale the input data
        scaled_data = self.scaler.transform(data.reshape(-1, 1))
        
        # Prepare the last sequence for prediction
        last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            next_pred = self.model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1, 0] = next_pred
        
        # Inverse transform to get original scale
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
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
        return self.predict(data, steps)