"""
Модуль для запуска всех моделей на всех данных
"""
from models.prophet_model import run_simple_prophet
from config.config import Config
import numpy as np
import pandas as pd


def run_all_models(series_id, values, dataset_name="M3", test_size=12):
    """
    Запуск всех моделей на одном временном ряде
    """
    results = {}
    
    # Запуск Prophet
    try:
        prophet_result = run_simple_prophet(series_id, values, dataset_name, test_size)
        if prophet_result and prophet_result['success']:
            results['Prophet'] = prophet_result
    except Exception as e:
        print(f"❌ Ошибка при запуске Prophet для {series_id}: {str(e)}")
    
    # Запуск ARIMA (если возможно)
    try:
        from models.arima_model import run_simple_arima
        arima_result = run_simple_arima(series_id, values, dataset_name, test_size)
        if arima_result and arima_result['success']:
            results['ARIMA'] = arima_result
    except ImportError:
        print("⚠️ ARIMA модель недоступна")
    except Exception as e:
        print(f"❌ Ошибка при запуске ARIMA для {series_id}: {str(e)}")
    
    # Запуск ETS (если возможно)
    try:
        from models.ets_model import run_simple_ets
        ets_result = run_simple_ets(series_id, values, dataset_name, test_size)
        if ets_result and ets_result['success']:
            results['ETS'] = ets_result
    except ImportError:
        print("⚠️ ETS модель недоступна")
    except Exception as e:
        print(f"❌ Ошибка при запуске ETS для {series_id}: {str(e)}")
    
    # Запуск LSTM (если возможно)
    try:
        from models.lstm_model import run_simple_lstm
        lstm_result = run_simple_lstm(series_id, values, dataset_name, test_size)
        if lstm_result and lstm_result['success']:
            results['LSTM'] = lstm_result
    except ImportError:
        print("⚠️ LSTM модель недоступна")
    except Exception as e:
        print(f"❌ Ошибка при запуске LSTM для {series_id}: {str(e)}")
    
    return results


def get_best_model_result(results):
    """
    Получение лучшего результата по метрике sMAPE
    """
    if not results:
        return None
    
    # Найти модель с минимальным sMAPE
    best_model = min(results.keys(), key=lambda k: results[k]['sMAPE'])
    best_result = results[best_model]
    best_result['Best_Model'] = best_model
    
    return best_result


def aggregate_model_results(results):
    """
    Агрегация результатов всех моделей для отчетности
    """
    if not results:
        return None
    
    aggregated = {
        'Series_ID': list(results.values())[0]['series_id'],
        'Dataset': list(results.values())[0]['dataset']
    }
    
    for model_name, result in results.items():
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