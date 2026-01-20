import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from utils.preprocessing import prepare_lstm_data, infer_period
from utils.metrics import calculate_metrics
from config.config import Config


def build_lstm_model(input_shape):
    """Построение LSTM модели для прогнозирования"""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def lstm_forecast(series, title, test_size=24, save_plots=True):
    """Прогнозирование с помощью LSTM"""
    try:
        from utils.visualization import setup_plot_style, plot_forecast_comparison

        # Подготовка данных
        seq_length = min(Config.LSTM_SEQUENCE_LENGTH, len(series) // 4)
        X, y, scaler, test = prepare_lstm_data(series, seq_length, test_size)
        train = series.iloc[:-test_size]

        # Построение и обучение модели
        model = build_lstm_model((X.shape[1], 1))
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(
            X, y,
            epochs=Config.LSTM_EPOCHS,
            batch_size=Config.LSTM_BATCH_SIZE,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=0
        )

        # Прогнозирование
        last_sequence = scaler.transform(train.values[-seq_length:].reshape(-1, 1))
        last_sequence = last_sequence.reshape(1, seq_length, 1)

        lstm_forecast = []
        for _ in range(test_size):
            next_pred = model.predict(last_sequence, verbose=0)
            lstm_forecast.append(next_pred[0, 0])

            # Подготовка новой последовательности
            new_sequence = np.zeros((1, seq_length, 1))
            if seq_length > 1:
                new_sequence[0, :seq_length - 1, 0] = last_sequence[0, 1:, 0]
            new_sequence[0, seq_length - 1, 0] = next_pred[0, 0]
            last_sequence = new_sequence

        # Обратное масштабирование
        lstm_forecast = np.array(lstm_forecast).reshape(-1, 1)
        lstm_forecast = scaler.inverse_transform(lstm_forecast).flatten()

        # Расчет метрик
        metrics = calculate_metrics(
            y_true=test.values,
            y_pred=lstm_forecast,
            y_train=train.values,
            m=infer_period(series)
        )
        metrics['Model'] = "LSTM"

        # Визуализация
        if save_plots:
            setup_plot_style()
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train.values, 'b-', label='Обучающие данные', linewidth=2)
            plt.plot(test.index, test.values, 'g-', label='Тестовые данные (факт)', linewidth=2)
            plt.plot(test.index, lstm_forecast, 'r--', label=f'LSTM прогноз (sMAPE={metrics["sMAPE (%)"]:.2f}%)',
                     linewidth=2.5)
            plt.title(f'Прогноз LSTM для {title}', fontsize=16)
            plt.xlabel('Дата', fontsize=14)
            plt.ylabel('Значение', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            save_path = Config.RESULTS_DIR / f'lstm_forecast_{title.replace(" ", "_")}.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

        return pd.Series(lstm_forecast, index=test.index), metrics

    except Exception as e:
        print(f"Ошибка при работе с LSTM для {title}: {e}")
        return None, None