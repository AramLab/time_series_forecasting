import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error (sMAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denom != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100

def mase(y_true, y_pred, y_train, m=12):
    """Mean Absolute Scaled Error (MASE)"""
    y_true, y_pred, y_train = map(np.array, [y_true, y_pred, y_train])
    if len(y_train) <= m:
        return np.nan
    naive_forecast = y_train[m:]
    naive_actual = y_train[:-m]
    scale = np.mean(np.abs(naive_actual - naive_forecast))
    if scale == 0:
        scale = 1.0
    return np.mean(np.abs(y_true - y_pred)) / scale

def calculate_metrics(y_true, y_pred, y_train, m):
    """Расчет всех метрик качества прогноза"""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'sMAPE (%)': smape(y_true, y_pred),
        'MASE': mase(y_true, y_pred, y_train, m)
    }