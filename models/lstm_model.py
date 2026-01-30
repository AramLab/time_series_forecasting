import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


class LSTMModel:
    def __init__(self, look_back=60, epochs=50, batch_size=32, validation_split=0.2, patience=5):
        """
        Initialize LSTM model with configuration parameters
        """
        self.look_back = look_back
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.patience = patience
        self.model = None
        self.scaler = None
    
    def prepare_data(self, data):
        """
        Prepare data for LSTM training
        """
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))

        X, y = [], []
        for i in range(self.look_back, len(scaled_data)):
            X.append(scaled_data[i-self.look_back:i, 0])
            y.append(scaled_data[i, 0])

        return np.array(X), np.array(y)

    def fit(self, data):
        """
        Train LSTM model on the provided data
        """
        X, y = self.prepare_data(data)

        # Reshape input to be [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # Create and compile the model
        self.model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)
        ])

        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        # Set up early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.patience,
            restore_best_weights=True
        )
        
        # Train the model
        self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping],
            verbose=0
        )
        
        return self

    def predict(self, data, steps):
        """
        Forecast future values using the trained LSTM model
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be fitted before making predictions")
            
        # Start with the last 'look_back' values from the original data
        last_sequence = data[-self.look_back:].copy()
        last_sequence_scaled = self.scaler.transform(last_sequence.reshape(-1, 1))

        predictions = []
        current_sequence = last_sequence_scaled.flatten().tolist()

        for _ in range(steps):
            # Reshape for prediction
            x_pred = np.array(current_sequence[-self.look_back:]).reshape(1, self.look_back, 1)
            
            # Predict next value
            next_pred = self.model.predict(x_pred, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Add prediction to sequence for next iteration
            current_sequence.append(next_pred)

        # Inverse transform the predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return predictions