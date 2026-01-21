from data.data_loader import DataLoader
from models.arima_model import auto_arima_forecast
from models.ets_model import auto_ets_forecast
from models.prophet_model import prophet_forecast
from models.ceemdan_models import ceemdan_combined_model
from models.lstm_model import lstm_forecast
from utils.visualization import plot_synthetic_series, plot_forecast_comparison, plot_model_comparison
from config.config import Config
import pandas as pd


class SyntheticDataAnalysis:
    def __init__(self):
        self.results = {}
        self.test_size = Config.TEST_SIZE

    def run_all_analyses(self):
        """Запуск анализа всех синтетических рядов"""
        print("=== АНАЛИЗ СИНТЕТИЧЕСКИХ ВРЕМЕННЫХ РЯДОВ ===")

        # Генерация синтетических данных
        synthetic_series = DataLoader.generate_synthetic_data()

        # Визуализация всех рядов
        plot_synthetic_series(
            synthetic_series,
            save_path=Config.RESULTS_DIR / 'all_series_comparison.png'
        )

        # Анализ каждого ряда
        for series_name, series in synthetic_series.items():
            print(f"\n{'=' * 80}")
            print(f"АНАЛИЗ РЯДА: {series.name}")
            print(f"{'=' * 80}")

            self.analyze_single_series(series, series_name)

        # Сравнение результатов
        self.compare_results()

    def analyze_single_series(self, series, series_name):
        """Анализ одного синтетического ряда всеми моделями"""
        forecasts = {}
        metrics_list = []

        print(f"\n--- ПРОГНОЗИРОВАНИЕ ARIMA ---")
        arima_forecast, arima_metrics = auto_arima_forecast(series, series.name, self.test_size)
        if arima_forecast is not None:
            forecasts['ARIMA'] = {'values': arima_forecast.values, 'metrics': arima_metrics}
            metrics_list.append(arima_metrics)

        print(f"\n--- ПРОГНОЗИРОВАНИЕ ETS ---")
        ets_forecast, ets_metrics = auto_ets_forecast(series, series.name, self.test_size)
        if ets_forecast is not None:
            forecasts['ETS'] = {'values': ets_forecast.values, 'metrics': ets_metrics}
            metrics_list.append(ets_metrics)

        print(f"\n--- ПРОГНОЗИРОВАНИЕ PROPHET ---")
        prophet_pred, prophet_metrics = prophet_forecast(series, series.name, self.test_size)
        if prophet_pred is not None:
            forecasts['Prophet'] = {'values': prophet_pred.values, 'metrics': prophet_metrics}
            metrics_list.append(prophet_metrics)

        print(f"\n--- ПРОГНОЗИРОВАНИЕ LSTM ---")
        lstm_pred, lstm_metrics = lstm_forecast(series, series.name, self.test_size)
        if lstm_pred is not None:
            forecasts['LSTM'] = {'values': lstm_pred.values, 'metrics': lstm_metrics}
            metrics_list.append(lstm_metrics)

        # CEEMDAN + ARIMA
        print(f"\n--- ПРОГНОЗИРОВАНИЕ CEEMDAN + ARIMA ---")
        ceemdan_arima_forecast, ceemdan_arima_metrics = ceemdan_combined_model(
            series,
            lambda s, title, test_size: auto_arima_forecast(s, title, test_size),
            series.name,
            test_size=self.test_size,
            model_name="CEEMDAN+ARIMA",
            save_plots=True
        )
        if ceemdan_arima_forecast is not None:
            forecasts['CEEMDAN+ARIMA'] = {'values': ceemdan_arima_forecast.values, 'metrics': ceemdan_arima_metrics}
            metrics_list.append(ceemdan_arima_metrics)
        else:
            print("⚠️ Прогноз CEEMDAN+ARIMA не выполнен")

        # CEEMDAN + ETS
        print(f"\n--- ПРОГНОЗИРОВАНИЕ CEEMDAN + ETS ---")
        ceemdan_ets_forecast, ceemdan_ets_metrics = ceemdan_combined_model(
            series,
            lambda s, title, test_size: auto_ets_forecast(s, title, test_size),
            series.name,
            test_size=self.test_size,
            model_name="CEEMDAN+ETS",
            save_plots=True
        )
        if ceemdan_ets_forecast is not None:
            forecasts['CEEMDAN+ETS'] = {'values': ceemdan_ets_forecast.values, 'metrics': ceemdan_ets_metrics}
            metrics_list.append(ceemdan_ets_metrics)
        else:
            print("⚠️ Прогноз CEEMDAN+ETS не выполнен")
        # Сравнение результатов
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list).set_index('Model')
            metrics_df = metrics_df[['RMSE', 'MAE', 'sMAPE (%)', 'MASE']]

            print(f"\nТаблица сравнения моделей по метрикам качества:")
            print(metrics_df.to_string(float_format="%.4f"))

            # Визуализация сравнения моделей
            plot_model_comparison(
                metrics_df,
                series.name,
                save_path=Config.RESULTS_DIR / f'model_comparison_{series_name}.png'
            )

            # Визуализация прогнозов
            train = series.iloc[:-self.test_size]
            test = series.iloc[-self.test_size:]

            plot_forecast_comparison(
                train, test, forecasts, series.name,
                save_path=Config.RESULTS_DIR / f'forecast_comparison_{series_name}.png'
            )

            self.results[series_name] = {
                'metrics': metrics_df,
                'forecasts': forecasts
            }

    def compare_results(self):
        """Сравнение результатов по всем синтетическим рядам"""
        if not self.results:
            print("Нет результатов для сравнения")
            return

        print(f"\n{'=' * 80}")
        print("СРАВНИТЕЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ ПО ВСЕМ СИНТЕТИЧЕСКИМ РЯДАМ")
        print(f"{'=' * 80}")

        # Сводная таблица по всем рядам
        summary_data = []

        for series_name, result in self.results.items():
            metrics_df = result['metrics']
            for model_name, row in metrics_df.iterrows():
                summary_data.append({
                    'Series': series_name,
                    'Model': model_name,
                    'sMAPE (%)': row['sMAPE (%)'],
                    'RMSE': row['RMSE'],
                    'MAE': row['MAE'],
                    'MASE': row['MASE']
                })

        summary_df = pd.DataFrame(summary_data)

        # Сохранение сводных результатов
        summary_df.to_csv(Config.RESULTS_DIR / 'synthetic_data_summary.csv', index=False)
        print(f"Сводные результаты сохранены в {Config.RESULTS_DIR / 'synthetic_data_summary.csv'}")

        # Лучшие модели для каждого ряда
        print("\nЛУЧШИЕ МОДЕЛИ ДЛЯ КАЖДОГО РЯДА:")
        print("-" * 60)

        for series_name in self.results.keys():
            series_results = summary_df[summary_df['Series'] == series_name]
            best_model = series_results.loc[series_results['sMAPE (%)'].idxmin()]
            print(f"{series_name}: {best_model['Model']} (sMAPE: {best_model['sMAPE (%)']:.2f}%)")