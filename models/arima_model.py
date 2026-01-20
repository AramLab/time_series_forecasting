import pmdarima as pm
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from utils.metrics import calculate_metrics
from utils.preprocessing import infer_period


def auto_arima_forecast(series, title, test_size=24, save_plots=True):
    """Автоматический подбор ARIMA с помощью pmdarima"""
    try:
        from utils.visualization import setup_plot_style

        # Разделение на обучающую и тестовую выборки
        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]

        # Определение периода сезонности
        m = infer_period(series)
        seasonality = m > 1

        # Автоматический подбор ARIMA
        model = pm.auto_arima(
            train,
            seasonal=seasonality,
            m=m if seasonality else 1,
            d=None,
            D=1 if seasonality else 0,
            max_p=3, max_q=3, max_P=2, max_Q=2,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )

        # Прогнозирование
        forecast = model.predict(n_periods=test_size)

        # Расчет метрик
        metrics = calculate_metrics(
            y_true=test.values,
            y_pred=forecast,
            y_train=train.values,
            m=m
        )
        metrics['Model'] = f"ARIMA{model.order}"

        # Визуализация
        if save_plots:
            setup_plot_style()
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train.values, 'b-', label='Обучающие данные', linewidth=2)
            plt.plot(test.index, test.values, 'g-', label='Тестовые данные (факт)', linewidth=2)
            plt.plot(test.index, forecast, 'r--', label=f'ARIMA прогноз (sMAPE={metrics["sMAPE (%)"]:.2f}%)',
                     linewidth=2.5)
            plt.title(f'Прогноз ARIMA для {title}', fontsize=16)
            plt.xlabel('Дата', fontsize=14)
            plt.ylabel('Значение', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            from config.config import Config
            save_path = Config.RESULTS_DIR / f'arima_forecast_{title.replace(" ", "_")}.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

        return pd.Series(forecast, index=test.index), metrics

    except Exception as e:
        print(f"Ошибка при работе с ARIMA для {title}: {e}")
        return None, None