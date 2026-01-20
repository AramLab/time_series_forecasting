import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils.preprocessing import infer_period
from utils.metrics import calculate_metrics
from config.config import Config


def safe_lstm_available():
    """Check if TensorFlow and Keras are available"""
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        return True, (tf, Sequential, LSTM, Dense, Dropout, EarlyStopping)
    except ImportError:
        return False, (None, None, None, None, None, None)


def simple_moving_average_forecast(series, test_size=24, window=5):
    """Simple moving average as fallback when LSTM is not available"""
    train = series.iloc[:-test_size]
    if len(train) < window:
        # If not enough data, use mean of available data
        last_values = np.full(test_size, np.mean(train.values))
    else:
        # Use moving average of last 'window' values for forecasting
        last_window = train.values[-window:]
        last_values = np.full(test_size, np.mean(last_window))
    return last_values


def build_lstm_model(input_shape, Sequential, LSTM, Dense, Dropout):
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


def prepare_simple_lstm_data(series, seq_length, test_size):
    """Simple preparation for LSTM data as fallback"""
    # Just return the scaled values without complex transformations
    from sklearn.preprocessing import MinMaxScaler
    
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
    
    # Create sequences manually
    X, y = [], []
    for i in range(seq_length, len(scaled_train)):
        X.append(scaled_train[i-seq_length:i])
        y.append(scaled_train[i])
    
    if len(X) == 0:
        return np.array([]).reshape(0, seq_length, 1), np.array([]), scaler, test
    
    X = np.array(X).reshape(len(X), seq_length, 1)
    y = np.array(y)
    
    return X, y, scaler, test


def lstm_forecast(series, title, test_size=24, save_plots=True):
    """Прогнозирование с помощью LSTM"""
    try:
        from utils.visualization import setup_plot_style
        
        # Check if TensorFlow is available
        is_available, (tf, Sequential, LSTM, Dense, Dropout, EarlyStopping) = safe_lstm_available()
        
        if not is_available:
            print(f"⚠️ TensorFlow недоступен, используем простой метод прогнозирования для {title}")
            
            # Use simple moving average as fallback
            forecast_values = simple_moving_average_forecast(series, test_size=test_size)
            test = series.iloc[-test_size:]
            train = series.iloc[:-test_size]
            
            # Calculate metrics
            metrics = calculate_metrics(
                y_true=test.values,
                y_pred=forecast_values,
                y_train=train.values,
                m=infer_period(series)
            )
            metrics['Model'] = "LSTM(fallback)"
            
            # Visualization
            if save_plots:
                setup_plot_style()
                plt.figure(figsize=(12, 6))
                plt.plot(train.index, train.values, 'b-', label='Обучающие данные', linewidth=2)
                plt.plot(test.index, test.values, 'g-', label='Тестовые данные (факт)', linewidth=2)
                plt.plot(test.index, forecast_values, 'r--', label=f'LSTM(fallback) прогноз (sMAPE={metrics["sMAPE (%)"]:.2f}%)',
                         linewidth=2.5)
                plt.title(f'Прогноз LSTM(fallback) для {title}', fontsize=16)
                plt.xlabel('Дата', fontsize=14)
                plt.ylabel('Значение', fontsize=14)
                plt.legend(fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()

                from config.config import Config
                save_path = Config.RESULTS_DIR / f'lstm_forecast_{title.replace(" ", "_")}.png'
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()

            return pd.Series(forecast_values, index=test.index), metrics
        
        # Original LSTM implementation if TensorFlow is available
        from utils.preprocessing import prepare_lstm_data
        
        # Подготовка данных
        seq_length = min(Config.LSTM_SEQUENCE_LENGTH, len(series) // 4)
        X, y, scaler, test = prepare_lstm_data(series, seq_length, test_size)
        train = series.iloc[:-test_size]

        if len(X) == 0:  # If not enough data for sequences
            print(f"⚠️ Недостаточно данных для создания LSTM-последовательностей, используем простой метод для {title}")
            forecast_values = simple_moving_average_forecast(series, test_size=test_size)
            test = series.iloc[-test_size:]
            
            metrics = calculate_metrics(
                y_true=test.values,
                y_pred=forecast_values,
                y_train=train.values,
                m=infer_period(series)
            )
            metrics['Model'] = "LSTM(fallback)"
            
            return pd.Series(forecast_values, index=test.index), metrics

        # Построение и обучение модели
        model = build_lstm_model((X.shape[1], 1), Sequential, LSTM, Dense, Dropout)
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

        lstm_forecast_list = []
        for _ in range(test_size):
            next_pred = model.predict(last_sequence, verbose=0)
            lstm_forecast_list.append(next_pred[0, 0])

            # Подготовка новой последовательности
            new_sequence = np.zeros((1, seq_length, 1))
            if seq_length > 1:
                new_sequence[0, :seq_length - 1, 0] = last_sequence[0, 1:, 0]
            new_sequence[0, seq_length - 1, 0] = next_pred[0, 0]
            last_sequence = new_sequence

        # Обратное масштабирование
        lstm_forecast_array = np.array(lstm_forecast_list).reshape(-1, 1)
        lstm_forecast_values = scaler.inverse_transform(lstm_forecast_array).flatten()

        # Расчет метрик
        metrics = calculate_metrics(
            y_true=test.values,
            y_pred=lstm_forecast_values,
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
            plt.plot(test.index, lstm_forecast_values, 'r--', label=f'LSTM прогноз (sMAPE={metrics["sMAPE (%)"]:.2f}%)',
                     linewidth=2.5)
            plt.title(f'Прогноз LSTM для {title}', fontsize=16)
            plt.xlabel('Дата', fontsize=14)
            plt.ylabel('Значение', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            from config.config import Config
            save_path = Config.RESULTS_DIR / f'lstm_forecast_{title.replace(" ", "_")}.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()

        return pd.Series(lstm_forecast_values, index=test.index), metrics

    except Exception as e:
        print(f"❌ ОШИБКА в LSTM модели для {title}: {e}")
        import traceback
        traceback.print_exc()
        
        # Final fallback
        try:
            test = series.iloc[-test_size:]
            forecast_values = simple_moving_average_forecast(series, test_size=test_size)
            
            metrics = calculate_metrics(
                y_true=test.values,
                y_pred=forecast_values,
                y_train=series.iloc[:-test_size].values,
                m=infer_period(series)
            )
            metrics['Model'] = "LSTM(fallback)"
            
            print("🔄 Используется наивный прогноз как фолбэк")
            return pd.Series(forecast_values, index=test.index), metrics
        except:
            return None, None