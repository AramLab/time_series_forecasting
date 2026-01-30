"""
Модуль для запуска всех моделей на всех данных
"""
from models.lstm_model import LSTMModel
from models.arima_model import ARIMAModel
from models.prophet_model import ProphetModel
from models.ets_model import ETSModel
from models.ceemdan_models import CEEMDANEnsembleModel, safe_import_ceemdan
from config.config import Config
import numpy as np
import pandas as pd


def run_all_models(series_id, values, dataset_name="M3", test_size=12):
    """
    Запуск всех моделей на одном временном ряде
    """
    results = {}
    
    # Convert values to appropriate format
    series = pd.Series(values, index=pd.date_range(start='2000-01-01', periods=len(values), freq='MS'))
    train_data = series.iloc[:-test_size].values
    test_data = series.iloc[-test_size:].values
    
    # Запуск LSTM
    try:
        lstm_model = LSTMModel(look_back=min(60, len(train_data)//2), epochs=50, batch_size=32)
        lstm_model.fit(train_data)
        lstm_forecast = lstm_model.predict(train_data, test_size)
        
        # Calculate metrics
        def calculate_smape(y_true, y_pred):
            epsilon = 1e-10
            return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + epsilon))
        
        smape = calculate_smape(test_data, lstm_forecast)
        rmse = np.sqrt(np.mean((test_data - lstm_forecast) ** 2))
        mae = np.mean(np.abs(test_data - lstm_forecast))
        
        results['LSTM'] = {
            'success': True,
            'series_id': series_id,
            'dataset': dataset_name,
            'forecast': lstm_forecast,
            'actual': test_data,
            'sMAPE': smape,
            'RMSE': rmse,
            'MAE': mae
        }
    except Exception as e:
        print(f"❌ Ошибка при запуске LSTM для {series_id}: {str(e)}")
    
    # Запуск ARIMA
    try:
        arima_model = ARIMAModel(max_p=5, max_q=5, max_d=2)
        arima_model.fit(train_data)
        arima_forecast = arima_model.predict(test_size)
        
        # Calculate metrics
        def calculate_smape(y_true, y_pred):
            epsilon = 1e-10
            return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + epsilon))
        
        smape = calculate_smape(test_data, arima_forecast)
        rmse = np.sqrt(np.mean((test_data - arima_forecast) ** 2))
        mae = np.mean(np.abs(test_data - arima_forecast))
        
        results['ARIMA'] = {
            'success': True,
            'series_id': series_id,
            'dataset': dataset_name,
            'forecast': arima_forecast,
            'actual': test_data,
            'sMAPE': smape,
            'RMSE': rmse,
            'MAE': mae
        }
    except Exception as e:
        print(f"❌ Ошибка при запуске ARIMA для {series_id}: {str(e)}")
    
    # Запуск Prophet
    try:
        prophet_model = ProphetModel(growth='linear', yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        prophet_model.fit(train_data, freq='MS')
        prophet_forecast = prophet_model.predict(test_size)
        
        # Calculate metrics
        def calculate_smape(y_true, y_pred):
            epsilon = 1e-10
            return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + epsilon))
        
        smape = calculate_smape(test_data, prophet_forecast)
        rmse = np.sqrt(np.mean((test_data - prophet_forecast) ** 2))
        mae = np.mean(np.abs(test_data - prophet_forecast))
        
        results['Prophet'] = {
            'success': True,
            'series_id': series_id,
            'dataset': dataset_name,
            'forecast': prophet_forecast,
            'actual': test_data,
            'sMAPE': smape,
            'RMSE': rmse,
            'MAE': mae
        }
    except Exception as e:
        print(f"❌ Ошибка при запуске Prophet для {series_id}: {str(e)}")
    
    # Запуск ETS
    try:
        ets_model = ETSModel(season_length=12, model='ZZZ')  # Assuming monthly data
        ets_model.fit(train_data, freq='MS')
        ets_forecast = ets_model.predict(test_size)
        
        # Calculate metrics
        def calculate_smape(y_true, y_pred):
            epsilon = 1e-10
            return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + epsilon))
        
        smape = calculate_smape(test_data, ets_forecast)
        rmse = np.sqrt(np.mean((test_data - ets_forecast) ** 2))
        mae = np.mean(np.abs(test_data - ets_forecast))
        
        results['ETS'] = {
            'success': True,
            'series_id': series_id,
            'dataset': dataset_name,
            'forecast': ets_forecast,
            'actual': test_data,
            'sMAPE': smape,
            'RMSE': rmse,
            'MAE': mae
        }
    except Exception as e:
        print(f"❌ Ошибка при запуске ETS для {series_id}: {str(e)}")
    
    # Запуск CEEMDAN+LSTM (если возможно)
    try:
        ceemdan_lstm_model = CEEMDANEnsembleModel(
            base_model_class=lambda: LSTMModel(look_back=min(30, len(train_data)//2), epochs=20, batch_size=16),
            trials=10,
            noise_width=0.05
        )
        ceemdan_lstm_model.fit(train_data)
        ceemdan_lstm_forecast = ceemdan_lstm_model.predict(test_size)
        
        # Calculate metrics
        def calculate_smape(y_true, y_pred):
            epsilon = 1e-10
            return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + epsilon))
        
        smape = calculate_smape(test_data, ceemdan_lstm_forecast)
        rmse = np.sqrt(np.mean((test_data - ceemdan_lstm_forecast) ** 2))
        mae = np.mean(np.abs(test_data - ceemdan_lstm_forecast))
        
        results['CEEMDAN+LSTM'] = {
            'success': True,
            'series_id': series_id,
            'dataset': dataset_name,
            'forecast': ceemdan_lstm_forecast,
            'actual': test_data,
            'sMAPE': smape,
            'RMSE': rmse,
            'MAE': mae
        }
    except Exception as e:
        print(f"❌ Ошибка при запуске CEEMDAN+LSTM для {series_id}: {str(e)}")
    
    # Запуск CEEMDAN+ARIMA (если возможно)
    try:
        ceemdan_arima_model = CEEMDANEnsembleModel(
            base_model_class=lambda: ARIMAModel(max_p=3, max_q=3, max_d=1),
            trials=10,
            noise_width=0.05
        )
        ceemdan_arima_model.fit(train_data)
        ceemdan_arima_forecast = ceemdan_arima_model.predict(test_size)
        
        # Calculate metrics
        def calculate_smape(y_true, y_pred):
            epsilon = 1e-10
            return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + epsilon))
        
        smape = calculate_smape(test_data, ceemdan_arima_forecast)
        rmse = np.sqrt(np.mean((test_data - ceemdan_arima_forecast) ** 2))
        mae = np.mean(np.abs(test_data - ceemdan_arima_forecast))
        
        results['CEEMDAN+ARIMA'] = {
            'success': True,
            'series_id': series_id,
            'dataset': dataset_name,
            'forecast': ceemdan_arima_forecast,
            'actual': test_data,
            'sMAPE': smape,
            'RMSE': rmse,
            'MAE': mae
        }
    except Exception as e:
        print(f"❌ Ошибка при запуске CEEMDAN+ARIMA для {series_id}: {str(e)}")
    
    return results


def get_best_model_result(results):
    """
    Получение лучшего результата по метрике sMAPE
    """
    if not results:
        return None
    
    # Filter out unsuccessful results
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_results:
        return None
    
    # Find model with minimum sMAPE
    best_model = min(successful_results.keys(), key=lambda k: successful_results[k]['sMAPE'])
    best_result = successful_results[best_model]
    best_result['Best_Model'] = best_model
    
    return best_result


def aggregate_model_results(results):
    """
    Агрегация результатов всех моделей для отчетности
    """
    if not results:
        return None
    
    # Filter out unsuccessful results
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if not successful_results:
        return None
    
    first_result = list(successful_results.values())[0]
    aggregated = {
        'Series_ID': first_result['series_id'],
        'Dataset': first_result['dataset']
    }
    
    for model_name, result in successful_results.items():
        aggregated[f'{model_name}_sMAPE'] = result.get('sMAPE', np.nan)
        aggregated[f'{model_name}_RMSE'] = result.get('RMSE', np.nan)
        aggregated[f'{model_name}_MAE'] = result.get('MAE', np.nan)
    
    # Добавляем лучшую модель
    best_result = get_best_model_result(results)
    if best_result:
        aggregated['Best_Model'] = best_result.get('Best_Model')
        aggregated['Best_sMAPE'] = best_result.get('sMAPE')
        aggregated['Best_RMSE'] = best_result.get('RMSE')
        aggregated['Best_MAE'] = best_result.get('MAE')
    
    return aggregated