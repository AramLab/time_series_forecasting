import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def infer_period(series):
    """Определение периода сезонности на основе частоты индекса"""
    try:
        if hasattr(series.index, 'freq') and series.index.freq is not None:
            if hasattr(series.index.freq, 'name'):
                freq_name = series.index.freq.name
            else:
                freq_name = str(series.index.freq)

            if 'M' in freq_name or 'MS' in freq_name:
                return 12
            elif 'Q' in freq_name:
                return 4
            elif 'W' in freq_name:
                return 52
            elif 'D' in freq_name:
                return 7
    except:
        pass

    # Если не удалось определить, используем эвристику на основе длины ряда
    if len(series) > 100:
        return 12  # месячные данные
    elif len(series) > 24:
        return 4  # квартальные данные
    return 1  # без сезонности


def prepare_lstm_data(series, seq_length, test_size):
    """Подготовка данных для LSTM модели"""
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train.values.reshape(-1, 1))

    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i - seq_length:i, 0])
        y.append(scaled_data[i, 0])

    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler, test