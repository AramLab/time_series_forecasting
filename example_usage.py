"""
Example usage of the Time Series Forecasting Project
Demonstrates all models and their usage
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from forecasting.models import LSTMModel, ARIMAModel, ETSModel, ProphetModel
from forecasting.ceemdan_models import CEEMDANLSTM, CEEMDANARIMA, CEEMDANETS, CEEMDANProphet
from forecasting.data import DataLoader, SyntheticGenerator
from forecasting.utils.metrics import calculate_metrics


def main():
    print("Time Series Forecasting Project - Example Usage")
    print("="*50)
    
    # Generate synthetic data
    print("1. Generating synthetic time series data...")
    generator = SyntheticGenerator()
    data = generator.generate_trend_seasonal_series(length=100, trend_slope=0.1, seasonal_period=12)
    print(f"Generated series with length: {len(data)}")
    
    # Split data into train and test
    train_data = data[:-12]  # Use last 12 points for testing
    test_data = data[-12:]
    
    print(f"Training data length: {len(train_data)}")
    print(f"Test data length: {len(test_data)}")
    print()
    
    # Test all models
    models_results = {}
    
    print("2. Testing individual models...")
    
    # LSTM Model
    print("   Testing LSTM...")
    try:
        lstm_model = LSTMModel(sequence_length=10, epochs=50)  # Reduced epochs for demo
        lstm_predictions = lstm_model.fit_predict(train_data, steps=12)
        lstm_metrics = calculate_metrics(test_data, lstm_predictions)
        models_results['LSTM'] = lstm_metrics
        print(f"      RMSE: {lstm_metrics['rmse']:.4f}, SMAPE: {lstm_metrics['smape']:.4f}%")
    except Exception as e:
        print(f"      Error with LSTM: {str(e)}")
    
    # ARIMA Model
    print("   Testing ARIMA...")
    try:
        arima_model = ARIMAModel()
        arima_predictions = arima_model.fit_predict(train_data, steps=12)
        arima_metrics = calculate_metrics(test_data, arima_predictions)
        models_results['ARIMA'] = arima_metrics
        print(f"      RMSE: {arima_metrics['rmse']:.4f}, SMAPE: {arima_metrics['smape']:.4f}%")
    except Exception as e:
        print(f"      Error with ARIMA: {str(e)}")
    
    # ETS Model
    print("   Testing ETS...")
    try:
        ets_model = ETSModel()
        ets_predictions = ets_model.fit_predict(train_data, steps=12)
        ets_metrics = calculate_metrics(test_data, ets_predictions)
        models_results['ETS'] = ets_metrics
        print(f"      RMSE: {ets_metrics['rmse']:.4f}, SMAPE: {ets_metrics['smape']:.4f}%")
    except Exception as e:
        print(f"      Error with ETS: {str(e)}")
    
    # Prophet Model
    print("   Testing Prophet...")
    try:
        prophet_model = ProphetModel()
        prophet_predictions = prophet_model.fit_predict(train_data, steps=12)
        prophet_metrics = calculate_metrics(test_data, prophet_predictions)
        models_results['Prophet'] = prophet_metrics
        print(f"      RMSE: {prophet_metrics['rmse']:.4f}, SMAPE: {prophet_metrics['smape']:.4f}%")
    except Exception as e:
        print(f"      Error with Prophet: {str(e)}")
    
    print()
    print("3. Testing CEEMDAN ensemble models...")
    
    # CEEMDAN+LSTM Model
    print("   Testing CEEMDAN+LSTM...")
    try:
        ceemdan_lstm = CEEMDANLSTM(ensemble_size=5, max_imfs=5)  # Reduced for demo
        ceemdan_lstm_predictions = ceemdan_lstm.fit_predict(train_data, steps=12)
        ceemdan_lstm_metrics = calculate_metrics(test_data, ceemdan_lstm_predictions)
        models_results['CEEMDAN+LSTM'] = ceemdan_lstm_metrics
        print(f"      RMSE: {ceemdan_lstm_metrics['rmse']:.4f}, SMAPE: {ceemdan_lstm_metrics['smape']:.4f}%")
    except Exception as e:
        print(f"      Error with CEEMDAN+LSTM: {str(e)}")
    
    # CEEMDAN+ARIMA Model
    print("   Testing CEEMDAN+ARIMA...")
    try:
        ceemdan_arima = CEEMDANARIMA(ensemble_size=5, max_imfs=5)  # Reduced for demo
        ceemdan_arima_predictions = ceemdan_arima.fit_predict(train_data, steps=12)
        ceemdan_arima_metrics = calculate_metrics(test_data, ceemdan_arima_predictions)
        models_results['CEEMDAN+ARIMA'] = ceemdan_arima_metrics
        print(f"      RMSE: {ceemdan_arima_metrics['rmse']:.4f}, SMAPE: {ceemdan_arima_metrics['smape']:.4f}%")
    except Exception as e:
        print(f"      Error with CEEMDAN+ARIMA: {str(e)}")
    
    # CEEMDAN+ETS Model
    print("   Testing CEEMDAN+ETS...")
    try:
        ceemdan_ets = CEEMDANETS(ensemble_size=5, max_imfs=5)  # Reduced for demo
        ceemdan_ets_predictions = ceemdan_ets.fit_predict(train_data, steps=12)
        ceemdan_ets_metrics = calculate_metrics(test_data, ceemdan_ets_predictions)
        models_results['CEEMDAN+ETS'] = ceemdan_ets_metrics
        print(f"      RMSE: {ceemdan_ets_metrics['rmse']:.4f}, SMAPE: {ceemdan_ets_metrics['smape']:.4f}%")
    except Exception as e:
        print(f"      Error with CEEMDAN+ETS: {str(e)}")
    
    # CEEMDAN+Prophet Model
    print("   Testing CEEMDAN+Prophet...")
    try:
        ceemdan_prophet = CEEMDANProphet(ensemble_size=5, max_imfs=5)  # Reduced for demo
        ceemdan_prophet_predictions = ceemdan_prophet.fit_predict(train_data, steps=12)
        ceemdan_prophet_metrics = calculate_metrics(test_data, ceemdan_prophet_predictions)
        models_results['CEEMDAN+Prophet'] = ceemdan_prophet_metrics
        print(f"      RMSE: {ceemdan_prophet_metrics['rmse']:.4f}, SMAPE: {ceemdan_prophet_metrics['smape']:.4f}%")
    except Exception as e:
        print(f"      Error with CEEMDAN+Prophet: {str(e)}")
    
    print()
    print("4. Model Comparison Results:")
    print("-" * 40)
    
    # Sort models by RMSE
    sorted_models = sorted(models_results.items(), key=lambda x: x[1]['rmse'])
    
    for model_name, metrics in sorted_models:
        print(f"{model_name:15s} | RMSE: {metrics['rmse']:6.4f} | SMAPE: {metrics['smape']:6.4f}% | MAE: {metrics['mae']:6.4f}")
    
    print()
    print("Example completed successfully!")


if __name__ == "__main__":
    main()