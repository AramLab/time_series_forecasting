import pandas as pd
import numpy as np
from pathlib import Path
from config.config import Config
from tqdm import tqdm


class DataLoader:
    @staticmethod
    def load_m3_data(group='Monthly', start_year=None, start_month=None, start_day=None):
        """Загрузка данных M3 с возможностью указания даты начала"""
        try:
            from datasetsforecast.m3 import M3
            print("Загрузка данных M3...")
            m3_data = M3.load(directory=str(Config.DATA_DIR), group=group)
            m3_df = m3_data[0] if isinstance(m3_data, tuple) else m3_data
            
            # Если указаны параметры даты, применяем их к индексу
            if start_year is not None or start_month is not None or start_day is not None:
                if start_year is None: start_year = 2000
                if start_month is None: start_month = 1
                if start_day is None: start_day = 1
                
                # Создаем новые даты на основе указанных параметров
                original_len = len(m3_df)
                new_dates = pd.date_range(
                    start=f'{start_year}-{start_month:02d}-{start_day:02d}',
                    periods=original_len,
                    freq='MS' if group == 'Monthly' else 'D'
                )
                
                # Обновляем индексы для соответствия новым датам
                if 'ds' in m3_df.columns:
                    m3_df['ds'] = new_dates[:len(m3_df)]
            
            print(f"M3 загружено: {len(m3_df)} записей, {m3_df['unique_id'].nunique()} рядов")
            return m3_df
        except ImportError as e:
            print(f"Ошибка при загрузке M3: {e}")
            return None

    @staticmethod
    def load_m4_data(group='Monthly', start_year=None, start_month=None, start_day=None):
        """Загрузка данных M4 с возможностью указания даты начала"""
        try:
            from datasetsforecast.m4 import M4
            print("Загрузка данных M4...")
            m4_data = M4.load(directory=str(Config.DATA_DIR), group=group)
            m4_df = m4_data[0] if isinstance(m4_data, tuple) else m4_data
            
            # Если указаны параметры даты, применяем их к индексу
            if start_year is not None or start_month is not None or start_day is not None:
                if start_year is None: start_year = 2000
                if start_month is None: start_month = 1
                if start_day is None: start_day = 1
                
                # Создаем новые даты на основе указанных параметров
                original_len = len(m4_df)
                new_dates = pd.date_range(
                    start=f'{start_year}-{start_month:02d}-{start_day:02d}',
                    periods=original_len,
                    freq='MS' if group == 'Monthly' else 'D'
                )
                
                # Обновляем индексы для соответствия новым датам
                if 'ds' in m4_df.columns:
                    m4_df['ds'] = new_dates[:len(m4_df)]
            
            print(f"M4 загружено: {len(m4_df)} записей, {m4_df['unique_id'].nunique()} рядов")
            return m4_df
        except ImportError as e:
            print(f"Ошибка при загрузке M4: {e}")
            return None

    @staticmethod
    def generate_synthetic_data(start_year=2000, start_month=1, start_day=1):
        """Генерация синтетических временных рядов с возможностью указания даты начала"""
        np.random.seed(Config.RANDOM_SEED)

        # 1. Ряд с трендом и сезонностью
        dates1 = pd.date_range(start=f'{start_year}-{start_month:02d}-{start_day:02d}', periods=240, freq='MS')
        trend1 = 0.5 * np.arange(len(dates1))
        seasonal1 = 20 * np.sin(2 * np.pi * dates1.month / 12)
        noise1 = np.random.normal(0, 5, len(dates1))
        trend_seasonal = pd.Series(100 + trend1 + seasonal1 + noise1, index=dates1, name='Тренд + сезонность')

        # 2. Ряд без тренда с сезонностью
        dates2 = pd.date_range(start=f'{start_year}-{start_month:02d}-{start_day:02d}', periods=240, freq='MS')
        seasonal2 = 15 * np.sin(2 * np.pi * dates2.month / 12)
        noise2 = np.random.normal(0, 3, len(dates2))
        no_trend_seasonal = pd.Series(100 + seasonal2 + noise2, index=dates2, name='Без тренда + сезонность')

        # 3. Ряд с трендом без сезонности
        dates3 = pd.date_range(start=f'{start_year}-{start_month:02d}-{start_day:02d}', periods=200, freq='D')
        trend3 = 0.3 * np.arange(len(dates3))
        noise3 = np.random.normal(0, 4, len(dates3))
        trend_no_seasonal = pd.Series(50 + trend3 + noise3, index=dates3, name='Тренд без сезонности')

        # 4. Ряд с сильным шумом
        dates4 = pd.date_range(start=f'{start_year}-{start_month:02d}-{start_day:02d}', periods=200, freq='D')
        noise4 = np.random.normal(0, 20, len(dates4))
        strong_noise = pd.Series(100 + noise4, index=dates4, name='Сильный шум')

        # 5. Ряд с трендом, сезонностью и высоким уровнем шума
        dates5 = pd.date_range(start=f'{start_year}-{start_month:02d}-{start_day:02d}', periods=240, freq='MS')
        trend5 = 0.4 * np.arange(len(dates5))
        seasonal5 = 25 * np.sin(2 * np.pi * dates5.month / 12)
        noise5 = np.random.normal(0, 15, len(dates5))
        trend_seasonal_high_noise = pd.Series(150 + trend5 + seasonal5 + noise5, index=dates5,
                                              name='Тренд + сезонность + шум')

        # 6. Ряд с сильным шумом и сезонностью
        dates6 = pd.date_range(start=f'{start_year}-{start_month:02d}-{start_day:02d}', periods=240, freq='MS')
        seasonal6 = 30 * np.sin(2 * np.pi * dates6.month / 12)
        noise6 = np.random.normal(0, 40, len(dates6))
        seasonal_high_noise = pd.Series(200 + seasonal6 + noise6, index=dates6, name='Сезонность + высокий шум')

        synthetic_series = {
            'trend_seasonal': trend_seasonal,
            'no_trend_seasonal': no_trend_seasonal,
            'trend_no_seasonal': trend_no_seasonal,
            'strong_noise': strong_noise,
            'trend_seasonal_high_noise': trend_seasonal_high_noise,
            'seasonal_high_noise': seasonal_high_noise
        }

        return synthetic_series

    @staticmethod
    def process_series_with_progress(series_dict, process_func):
        """
        Обработка временных рядов с отображением прогресс-бара
        
        Args:
            series_dict: Словарь с временными рядами
            process_func: Функция для обработки каждого ряда
        """
        processed_series = {}
        
        # Создаем прогресс-бар
        pbar = tqdm(
            iterable=series_dict.items(),
            total=len(series_dict),
            desc="Обработка временных рядов",
            unit="ряд"
        )
        
        for name, series in pbar:
            # Обновляем описание прогресс-бара
            pbar.set_postfix({"Ряд": name})
            
            # Применяем функцию обработки к текущему ряду
            processed_series[name] = process_func(series)
        
        return processed_series