import matplotlib.pyplot as plt
from prophet import Prophet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.metrics import calculate_metrics
from utils.preprocessing import infer_period
from config.config import Config

def run_simple_prophet(series_id, values, dataset_name="M3", test_size=12):
    """
    Простая и надежная функция прогнозирования с Prophet
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

        # 3. Создание искусственных дат для Prophet
        start_date = pd.Timestamp('2000-01-01')
        train_dates = pd.date_range(start=start_date, periods=len(train_values), freq='MS')
        test_dates = pd.date_range(start=train_dates[-1] + pd.Timedelta(days=31), periods=len(test_values), freq='MS')

        # 4. Подготовка данных для Prophet
        train_df = pd.DataFrame({'ds': train_dates, 'y': train_values})

        # 5. Создание модели Prophet с безопасными параметрами
        model = Prophet(
            growth='linear',
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            seasonality_mode='additive',
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_range=0.8,
            interval_width=0.95
        )

        model.fit(train_df)

        # 6. Прогнозирование
        future_df = pd.DataFrame({'ds': test_dates})
        forecast = model.predict(future_df)
        forecast_values = forecast['yhat'].values

        # 7. Расчет метрик
        def calculate_smape(y_true, y_pred):
            epsilon = 1e-10
            return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + epsilon))

        smape_val = calculate_smape(test_values, forecast_values)
        rmse = np.sqrt(mean_squared_error(test_values, forecast_values))
        mae = mean_absolute_error(test_values, forecast_values)

        # 8. Простая визуализация
        setup_plot_style()
        plt.figure(figsize=(12, 6))
        train_idx = range(len(train_values))
        test_idx = range(len(train_values), len(train_values) + len(test_values))

        plt.plot(train_idx, train_values, 'b-', linewidth=2, label='Обучающие данные', alpha=0.7)
        plt.plot(test_idx, test_values, 'g-', linewidth=2, label='Фактические значения', alpha=0.7)
        plt.plot(test_idx, forecast_values, 'r--', linewidth=2.5,
                 label=f'Прогноз Prophet (sMAPE={smape_val:.2f}%)', alpha=0.9)

        if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
            plt.fill_between(test_idx,
                             forecast['yhat_lower'].values,
                             forecast['yhat_upper'].values,
                             color='red', alpha=0.2, label='95% доверительный интервал')

        plt.title(f'Прогноз Prophet для {dataset_name}: {series_id}', fontsize=14)
        plt.xlabel('Период', fontsize=12)
        plt.ylabel('Значение', fontsize=12)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = Config.RESULTS_DIR / f'prophet_{dataset_name}_{series_id}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

        # 9. Возвращаем результаты
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


def prophet_forecast(series, title, test_size=24, auto_tune=False):
    """Прогнозирование с помощью Prophet"""
    try:
        from prophet import Prophet

        # Разделение на обучающую и тестовую выборки
        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]

        # Подготовка данных для Prophet
        df = pd.DataFrame({'ds': train.index, 'y': train.values})

        # Определение сезонности
        m = infer_period(series)
        has_yearly_seasonality = (m == 12)
        has_weekly_seasonality = (m == 7)

        # Создание модели
        seasonality_mode = 'multiplicative' if np.mean(series) > 50 else 'additive'
        model = Prophet(
            yearly_seasonality=has_yearly_seasonality,
            weekly_seasonality=has_weekly_seasonality,
            daily_seasonality=False,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=0.05,
            interval_width=0.95
        )

        # Обучение модели
        model.fit(df)

        # Создание будущих дат для прогноза
        last_date = train.index[-1]
        freq = series.index.freqstr if hasattr(series.index, 'freq') else pd.infer_freq(series.index)
        if freq is None:
            freq = 'MS' if m == 12 else 'D'

        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=test_size, freq=freq)
        future_df = pd.DataFrame({'ds': future_dates})

        # Прогнозирование
        forecast = model.predict(future_df)
        forecast_values = forecast['yhat'].values

        # Расчет метрик
        metrics = calculate_metrics(
            y_true=test.values,
            y_pred=forecast_values,
            y_train=train.values,
            m=m
        )
        metrics['Model'] = 'Prophet'

        return pd.Series(forecast_values, index=test.index), metrics

    except Exception as e:
        print(f"Ошибка при работе с Prophet для {title}: {e}")
        return None, None