from matplotlib import pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoETS
import pandas as pd
import numpy as np
from utils.metrics import calculate_metrics
from utils.preprocessing import infer_period


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