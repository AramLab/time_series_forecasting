from matplotlib import pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoETS
import pandas as pd
import numpy as np
from utils.metrics import calculate_metrics
from utils.preprocessing import infer_period
from config.config import Config
from sklearn.metrics import mean_squared_error, mean_absolute_error


def auto_ets_forecast(series, title, test_size=24, save_plots=True):
    """Автоматический подбор ETS с помощью statsforecast"""
    try:
        from utils.visualization import setup_plot_style

        # Разделение на обучающую и тестовую выборки
        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]

        # Преобразование данных для statsforecast
        df = pd.DataFrame({
            'unique_id': 1,
            'ds': train.index,
            'y': train.values
        })

        # Определение периода сезонности
        m = infer_period(series)
        seasonality = m > 1

        # Создание и обучение модели
        models = [
            AutoETS(season_length=m if seasonality else 1, model='ZZZ')
        ]
        sf = StatsForecast(models=models, freq=series.index.freqstr, n_jobs=-1)
        sf.fit(df)

        # Прогнозирование
        forecast_df = sf.predict(h=test_size)
        forecast = forecast_df[f'AutoETS'].values

        # Расчет метрик
        metrics = calculate_metrics(
            y_true=test.values,
            y_pred=forecast,
            y_train=train.values,
            m=m
        )
        metrics['Model'] = "AutoETS"

        # Визуализация
        if save_plots:
            setup_plot_style()
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train.values, 'b-', label='Обучающие данные', linewidth=2)
            plt.plot(test.index, test.values, 'g-', label='Тестовые данные (факт)', linewidth=2)
            plt.plot(test.index, forecast, 'r--', label=f'ETS прогноз (sMAPE={metrics["sMAPE (%)"]:.2f}%)',
                     linewidth=2.5)
            plt.title(f'Прогноз ETS для {title}', fontsize=16)
            plt.xlabel('Дата', fontsize=14)
            plt.ylabel('Значение', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            from config.config import Config
            save_path = Config.RESULTS_DIR / f'ets_forecast_{title.replace(" ", "_")}.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

        return pd.Series(forecast, index=test.index), metrics

    except Exception as e:
        print(f"Ошибка при работе с ETS для {title}: {e}")
        return None, None


def run_simple_ets(series_id, values, dataset_name="M3", test_size=12):
    """
    Простая и надежная функция прогнозирования с ETS
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

        # 3. Подготовка данных для ETS
        dates = pd.date_range(start='2000-01-01', periods=len(values), freq='MS')
        series = pd.Series(values, index=dates)

        # 4. Запуск ETS
        forecast_series, metrics = auto_ets_forecast(
            series=series,
            title=f"{dataset_name}: {series_id}",
            test_size=test_size
        )

        if forecast_series is None or metrics is None:
            print(f"⚠ Не удалось получить прогноз для {series_id}")
            return None

        forecast_values = forecast_series.values
        test_values = series[train_size:].values

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
                 label=f'Прогноз ETS (sMAPE={smape_val:.2f}%)', alpha=0.9)

        plt.title(f'Прогноз ETS для {dataset_name}: {series_id}', fontsize=14)
        plt.xlabel('Период', fontsize=12)
        plt.ylabel('Значение', fontsize=12)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = Config.RESULTS_DIR / f'ets_{dataset_name}_{series_id}.png'
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