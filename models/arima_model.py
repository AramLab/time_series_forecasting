import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from utils.metrics import calculate_metrics
from utils.preprocessing import infer_period
from config.config import Config


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


def auto_arima_forecast(series, title, test_size=24, save_plots=True, max_p=5, max_q=5, max_d=2):
    """ARIMA forecast using statsmodels with automatic order selection"""
    try:
        from utils.visualization import setup_plot_style
        
        # Prepare data
        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]
        
        if len(train) < 10:
            raise ValueError(f"Training set too small: {len(train)}. Need at least 10 points.")
        
        # Handle missing values
        train = train.dropna()
        test = test.dropna()
        
        if len(train) < 10:
            raise ValueError(f"Not enough valid training data after dropping NaNs: {len(train)}")
        
        # Determine differencing order (d)
        d = 0
        if not check_stationarity(train):
            # Try differencing
            temp_diff_1 = train.diff().dropna()
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
        best_model = None
        
        # Limit search space for smaller datasets
        max_p_actual = min(max_p, max(1, len(train)//4))  # Prevent overparameterization
        max_q_actual = min(max_q, max(1, len(train)//4))
        
        for p in range(0, max_p_actual + 1):
            for q in range(0, max_q_actual + 1):
                try:
                    model = ARIMA(train, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    if fitted_model.aic < best_aic:
                        best_aic = fitted_model.aic
                        best_order = (p, d, q)
                        best_model = fitted_model
                except:
                    continue
        
        if best_model is None:
            # If no model fits, try with minimal parameters
            d = min(d, 1)  # Limit differencing
            for p in range(0, min(2, max_p_actual + 1)):
                for q in range(0, min(2, max_q_actual + 1)):
                    try:
                        model = ARIMA(train, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            best_model = fitted_model
                        break
                    except:
                        continue
                if best_model is not None:
                    break
        
        if best_model is None:
            # Ultimate fallback: use simple method
            # Return naive forecast
            last_values = train.values[-test_size:] if len(train) >= test_size else np.full(test_size, np.mean(train.values))
            forecast_values = np.resize(last_values, test_size)
            
            # Calculate metrics
            metrics = calculate_metrics(
                y_true=test.values[:len(forecast_values)],
                y_pred=forecast_values,
                y_train=train.values,
                m=infer_period(series)
            )
            metrics['Model'] = 'ARIMA(naive)'
            metrics['Order'] = 'N/A'
            
            print(f"⚠️ ARIMA: Used naive forecast due to fitting issues. Order: N/A")
            return pd.Series(forecast_values, index=test.index[:len(forecast_values)]), metrics
        
        # Generate forecast
        forecast_result = best_model.forecast(steps=test_size)
        forecast_values = forecast_result.values
        
        # Handle cases where forecast returns fewer values than expected
        if len(forecast_values) < test_size:
            # Extend with last forecasted value
            extension = np.full(test_size - len(forecast_values), forecast_values[-1])
            forecast_values = np.concatenate([forecast_values, extension])
        elif len(forecast_values) > test_size:
            forecast_values = forecast_values[:test_size]
        
        # Calculate metrics
        metrics = calculate_metrics(
            y_true=test.values[:len(forecast_values)],
            y_pred=forecast_values,
            y_train=train.values,
            m=infer_period(series)
        )
        metrics['Model'] = 'ARIMA'
        metrics['Order'] = str(best_order)
        
        # Visualization
        if save_plots:
            setup_plot_style()
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train.values, 'b-', label='Обучающие данные', linewidth=2)
            plt.plot(test.index[:len(forecast_values)], test.values[:len(forecast_values)],
                     'g-', label='Тестовые данные (факт)', linewidth=2)
            plt.plot(test.index[:len(forecast_values)], forecast_values,
                     'r--', label=f'ARIMA прогноз (sMAPE={metrics["sMAPE (%)"]:.2f}%)', linewidth=2.5)
            plt.title(f'Прогноз ARIMA({best_order[0]},{best_order[1]},{best_order[2]}) для {title}', fontsize=16)
            plt.xlabel('Дата', fontsize=14)
            plt.ylabel('Значение', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            from config.config import Config
            safe_title = title.replace(" ", "_").replace("+", "_").replace("/", "_")
            save_path = Config.RESULTS_DIR / f'arima_forecast_{safe_title}.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"💾 График ARIMA сохранен: {save_path}")
        
        print(f"✅ ARIMA прогноз успешно выполнен для {title}")
        print(f"📊 Параметры: ARIMA{best_order}, RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, sMAPE={metrics['sMAPE (%)']:.2f}%")
        
        return pd.Series(forecast_values, index=test.index[:len(forecast_values)]), metrics
    
    except Exception as e:
        print(f"❌ ОШИБКА в ARIMA модели для {title}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback: naive forecast
        try:
            train = series.iloc[:-test_size]
            naive_forecast = np.full(test_size, np.mean(train.values[-min(5, len(train)):]))
            
            metrics = calculate_metrics(
                y_true=series.iloc[-test_size:].values,
                y_pred=naive_forecast,
                y_train=series.iloc[:-test_size].values,
                m=infer_period(series)
            )
            metrics['Model'] = 'ARIMA(naive)'
            metrics['Order'] = 'N/A'
            
            print("🔄 Используется наивный прогноз как фолбэк")
            return pd.Series(naive_forecast, index=series.iloc[-test_size:].index), metrics
        except:
            # Double fallback
            naive_forecast = np.full(test_size, np.mean(series.values[-10:]))
            return pd.Series(naive_forecast, index=pd.date_range(start=series.index[-1], periods=test_size+1, freq=pd.infer_freq(series.index))[1:]), None


def run_simple_arima(series_id, values, dataset_name="M3", test_size=12):
    """
    Простая и надежная функция прогнозирования с ARIMA
    """
    try:
        from utils.visualization import setup_plot_style

        # 1. Проверка данных
        if len(values) < test_size * 2:
            print(f"⚠ Слишком короткий ряд: {len(values)} < {test_size * 2}")
            return None

        # 2. Разделение данных
        train_size = len(values) - test_size
        train_values = values[:train_size]
        test_values = values[train_size:]

        # 3. Подготовка данных для ARIMA
        # Создание индекса с правильной частотой
        dates = pd.date_range(start='2000-01-01', periods=len(values), freq='MS')
        series = pd.Series(values, index=dates)
        train_series = series[:train_size]
        test_series = series[train_size:]

        # 4. Запуск ARIMA
        forecast_series, metrics = auto_arima_forecast(
            series=series,
            title=f"{dataset_name}: {series_id}",
            test_size=test_size
        )

        if forecast_series is None or metrics is None:
            print(f"⚠ Не удалось получить прогноз для {series_id}")
            return None

        forecast_values = forecast_series.values
        test_values = test_series.values

        # 5. Расчет метрик
        def calculate_smape(y_true, y_pred):
            epsilon = 1e-10
            return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + epsilon))

        # Используем метрики из результата или рассчитываем заново
        smape_val = metrics.get('sMAPE (%)', calculate_smape(test_values, forecast_values))
        rmse = metrics.get('RMSE', np.sqrt(mean_squared_error(test_values, forecast_values)))
        mae = metrics.get('MAE', mean_absolute_error(test_values, forecast_values))

        # 6. Простая визуализация
        setup_plot_style()
        plt.figure(figsize=(12, 6))
        train_idx = range(len(train_values))
        test_idx = range(len(train_values), len(train_values) + len(test_values))

        plt.plot(train_idx, train_values, 'b-', linewidth=2, label='Обучающие данные', alpha=0.7)
        plt.plot(test_idx, test_values, 'g-', linewidth=2, label='Фактические значения', alpha=0.7)
        plt.plot(test_idx, forecast_values, 'r--', linewidth=2.5,
                 label=f'Прогноз ARIMA (sMAPE={smape_val:.2f}%)', alpha=0.9)

        plt.title(f'Прогноз ARIMA для {dataset_name}: {series_id}', fontsize=14)
        plt.xlabel('Период', fontsize=12)
        plt.ylabel('Значение', fontsize=12)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = Config.RESULTS_DIR / f'arima_{dataset_name}_{series_id}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

        # 7. Возвращаем результаты
        return {
            'series_id': series_id,
            'dataset': dataset_name,
            'forecast': forecast_values,
            'actual': test_values,
            'sMAPE': smape_val,
            'RMSE': rmse,
            'MAE': mae,
            'success': True
        }

    except Exception as e:
        print(f"❌ Ошибка при прогнозировании {series_id}: {str(e)}")
        return {
            'series_id': series_id,
            'dataset': dataset_name,
            'error': str(e),
            'success': False
        }