#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Тестирование обновленного DataLoader с параметрами даты и прогресс-баром
"""

import pandas as pd
from data.data_loader import DataLoader


def sample_processing_function(series):
    """Пример функции для обработки временного ряда"""
    # Имитируем какую-то обработку
    processed = series.copy()
    # Например, нормализация
    mean_val = processed.mean()
    std_val = processed.std()
    if std_val != 0:
        processed = (processed - mean_val) / std_val
    else:
        processed = processed - mean_val
    
    # Добавляем небольшую задержку, чтобы прогресс-бар был заметен
    import time
    time.sleep(0.1)
    
    return processed


def main():
    print("=== Тестирование DataLoader с новыми параметрами ===\n")
    
    # Тест 1: Генерация синтетических данных с заданной датой начала
    print("1. Генерация синтетических данных с датой начала 2020-06-15:")
    synthetic_data = DataLoader.generate_synthetic_data(start_year=2020, start_month=6, start_day=15)
    
    for name, series in synthetic_data.items():
        print(f"   {name}: {len(series)} точек, период с {series.index[0]} по {series.index[-1]}")
    
    print("\n2. Обработка синтетических данных с прогресс-баром:")
    # Обработка рядов с использованием прогресс-бара
    processed_data = DataLoader.process_series_with_progress(synthetic_data, sample_processing_function)
    
    print("\n   Обработка завершена!")
    
    # Тест 2: Загрузка M3 данных (если доступны) с параметрами даты
    print("\n3. Загрузка M3 данных с датой начала 2021-01-01:")
    m3_data = DataLoader.load_m3_data(group='Monthly', start_year=2021, start_month=1, start_day=1)
    if m3_data is not None:
        print(f"   Успешно загружено M3 данных: {len(m3_data)} записей")
        if 'ds' in m3_data.columns:
            print(f"   Диапазон дат: от {m3_data['ds'].min()} до {m3_data['ds'].max()}")
    else:
        print("   Данные M3 недоступны (возможно, требуется установка datasetsforecast)")
    
    # Тест 3: Загрузка M4 данных (если доступны) с параметрами даты
    print("\n4. Загрузка M4 данных с датой начала 2022-03-10:")
    m4_data = DataLoader.load_m4_data(group='Monthly', start_year=2022, start_month=3, start_day=10)
    if m4_data is not None:
        print(f"   Успешно загружено M4 данных: {len(m4_data)} записей")
        if 'ds' in m4_data.columns:
            print(f"   Диапазон дат: от {m4_data['ds'].min()} до {m4_data['ds'].max()}")
    else:
        print("   Данные M4 недоступны (возможно, требуется установка datasetsforecast)")


if __name__ == "__main__":
    main()