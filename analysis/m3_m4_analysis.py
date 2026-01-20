from data.data_loader import DataLoader
from models.prophet_model import run_simple_prophet
from utils.visualization import plot_aggregated_results
from config.config import Config
import pandas as pd
import numpy as np


class M3M4Analysis:
    def __init__(self, max_series_per_dataset=10):
        self.max_series_per_dataset = max_series_per_dataset
        self.results_m3 = []
        self.results_m4 = []

    def run_analysis(self):
        """Запуск анализа данных M3 и M4"""
        print("=== АНАЛИЗ ДАННЫХ M3 И M4 ===")

        # Загрузка данных
        m3_df = DataLoader.load_m3_data()
        m4_df = DataLoader.load_m4_data()

        if m3_df is None or m4_df is None:
            print("Ошибка при загрузке данных. Анализ невозможен.")
            return

        # Анализ M3
        self.analyze_dataset(m3_df, "M3")

        # Анализ M4
        self.analyze_dataset(m4_df, "M4")

        # Агрегированный анализ
        self.aggregate_results()

    def analyze_dataset(self, df, dataset_name):
        """Анализ одного набора данных (M3 или M4)"""
        print(f"\n{'=' * 80}")
        print(f"АНАЛИЗ ДАННЫХ {dataset_name}")
        print(f"{'=' * 80}")

        # Получение уникальных ID рядов
        series_ids = df['unique_id'].unique()

        if self.max_series_per_dataset > 0 and self.max_series_per_dataset < len(series_ids):
            series_ids = series_ids[:self.max_series_per_dataset]
            print(f"  Анализ ограничен первыми {self.max_series_per_dataset} рядами")
        else:
            print(f"  Анализ всех {len(series_ids)} рядов")

        # Анализ каждого ряда
        total = len(series_ids)
        for i, series_id in enumerate(series_ids, 1):
            print(f"\n▶️ [{i}/{total}] Анализ ряда {dataset_name}-{series_id}")
            print("-" * 40)

            # Фильтрация данных для текущего ряда
            series_data = df[df['unique_id'] == series_id].sort_values('ds')

            # Проверка длины ряда
            if len(series_data) < Config.TEST_SIZE * 2:
                print(
                    f"⚠️  Ряд слишком короткий для анализа (требуется минимум {Config.TEST_SIZE * 2} точек, есть {len(series_data)}). Пропускаем.")
                continue

            # Извлечение значений
            values = series_data['y'].values

            # Прогнозирование с помощью Prophet
            result = run_simple_prophet(series_id, values, dataset_name, Config.TEST_SIZE)

            if result and result['success']:
                # Сохранение результатов
                result_data = {
                    'Dataset': dataset_name,
                    'Series_ID': series_id,
                    'sMAPE': result['sMAPE'],
                    'RMSE': result['RMSE'],
                    'MAE': result['MAE'],
                    'Length': len(values),
                    'Mean': np.mean(values),
                    'Std': np.std(values)
                }

                if dataset_name == "M3":
                    self.results_m3.append(result_data)
                else:
                    self.results_m4.append(result_data)

                print(f"✅ Успешно: sMAPE = {result['sMAPE']:.2f}%")
            else:
                print(f"❌ Не удалось выполнить прогноз")

    def aggregate_results(self):
        """Агрегация и анализ результатов"""
        print(f"\n{'=' * 80}")
        print("АГРЕГИРОВАННЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
        print(f"{'=' * 80}")

        # Создание DataFrame с результатами
        if self.results_m3:
            summary_m3 = pd.DataFrame(self.results_m3)
            summary_m3.to_csv(Config.RESULTS_DIR / 'm3_results.csv', index=False)
            print(f"✅ Результаты M3 сохранены в {Config.RESULTS_DIR / 'm3_results.csv'}")

        if self.results_m4:
            summary_m4 = pd.DataFrame(self.results_m4)
            summary_m4.to_csv(Config.RESULTS_DIR / 'm4_results.csv', index=False)
            print(f"✅ Результаты M4 сохранены в {Config.RESULTS_DIR / 'm4_results.csv'}")

        # Статистический анализ
        if self.results_m3 and self.results_m4:
            print(f"\n{'=' * 60}")
            print("СРАВНИТЕЛЬНАЯ СТАТИСТИКА M3 И M4")
            print(f"{'=' * 60}")

            summary_m3 = pd.DataFrame(self.results_m3)
            summary_m4 = pd.DataFrame(self.results_m4)

            # Средние значения по датасетам
            print(f"\nСРЕДНИЕ ЗНАЧЕНИЯ ПО ДАТАСЕТАМ:")
            print("-" * 40)

            for dataset_name, df in [('M3', summary_m3), ('M4', summary_m4)]:
                print(f"\n{dataset_name}:")
                print(f"  Средний sMAPE: {df['sMAPE'].mean():.2f}%")
                print(f"  Средний RMSE: {df['RMSE'].mean():.2f}")
                print(f"  Средний MAE: {df['MAE'].mean():.2f}")
                print(f"  Количество рядов: {len(df)}")

            # Визуализация результатов
            plot_aggregated_results(summary_m3, summary_m4, str(Config.RESULTS_DIR))
            print("✅ Графики агрегированных результатов сохранены")