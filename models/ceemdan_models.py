import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
from utils.metrics import calculate_metrics
from utils.preprocessing import infer_period


def safe_import_ceemdan():
    """Безопасный импорт CEEMDAN с обработкой различных версий PyEMD"""
    try:
        # Принудительно импортируем EMD как класс до импорта CEEMDAN
        # Это исправляет ошибку 'module' object is not callable
        from PyEMD.EMD import EMD as EMD_Class
        
        # Загружаем CEEMDAN напрямую
        from PyEMD.CEEMDAN import CEEMDAN as CEEMDAN_Class
        
        # Убедимся, что EMD доступен как вызываемый класс в PyEMD
        import PyEMD
        if not callable(PyEMD.EMD):
            PyEMD.EMD = EMD_Class
            
        print("✅ CEEMDAN успешно импортирован из PyEMD.CEEMDAN")
        return CEEMDAN_Class
    except ImportError:
        try:
            # Альтернативный путь импорта
            import PyEMD
            # Сначала убедимся, что EMD правильно импортирован как класс
            try:
                from PyEMD.EMD import EMD as EMD_Class
                # Заменяем EMD в PyEMD, если он является модулем
                if hasattr(PyEMD, 'EMD') and not callable(PyEMD.EMD):
                    PyEMD.EMD = EMD_Class
            except ImportError:
                pass  # EMD может быть уже доступен как класс
            
            # Пытаемся импортировать CEEMDAN напрямую
            from PyEMD import CEEMDAN as CEEMDAN_Class
            print("✅ CEEMDAN успешно импортирован из PyEMD")
            return CEEMDAN_Class
        except Exception as e:
            print(f"❌ Не удалось импортировать CEEMDAN: {e}")
            return None


def ceemdan_combined_model(series, base_model_fn, title, test_size=24, model_name="CEEMDAN+X", save_plots=True):
    """Комбинированная модель CEEMDAN + базовая модель"""
    try:
        from utils.visualization import setup_plot_style

        # Прежде чем использовать CEEMDAN, нужно убедиться, что EMD правильно импортирован
        # как класс, а не модуль, чтобы внутренние вызовы CEEMDAN работали корректно
        try:
            # Импортируем EMD как класс и устанавливаем его в PyEMD
            from PyEMD.EMD import EMD as EMD_Class
            import PyEMD
            if hasattr(PyEMD, 'EMD') and not callable(PyEMD.EMD):
                PyEMD.EMD = EMD_Class
        except ImportError:
            # Если прямой импорт EMD как класс не работает, пробуем альтернативный путь
            try:
                import PyEMD
                if hasattr(PyEMD, 'EMD') and not callable(PyEMD.EMD):
                    # Пытаемся получить EMD класс другим способом
                    from PyEMD import EMD as EMD_Class
                    PyEMD.EMD = EMD_Class
            except ImportError:
                pass  # Если ничего не работает, продолжаем с надеждой, что CEEMDAN будет работать

        # Безопасное получение класса CEEMDAN
        CEEMDAN_Class = safe_import_ceemdan()
        if CEEMDAN_Class is None:
            print(f"❌ CEEMDAN недоступен. Пропускаем {model_name} прогнозирование.")
            return None, None

        # Разделение на обучающую и тестовую выборки
        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]

        # CEEMDAN декомпозиция - ИСПРАВЛЕНО: правильное создание экземпляра
        print("🔍 Выполняем CEEMDAN декомпозицию...")
        print(f"📊 Данные для декомпозиции: {len(train)} точек")

        # Создаем экземпляр CEEMDAN
        ceemdan_instance = CEEMDAN_Class(trials=20, noise_width=0.05)
        print("✅ CEEMDAN экземпляр успешно создан")

        # Выполняем декомпозицию
        imfs = ceemdan_instance(train.values.astype(float))
        print(f"✅ Получено IMF компонент: {len(imfs)}")

        # Прогнозирование каждой IMF
        imf_forecasts = []
        successful_imfs = 0

        for i, imf in enumerate(imfs):
            print(f"📈 Прогнозирование IMF {i + 1}/{len(imfs)}...")
            try:
                # Создание временного ряда для IMF
                imf_series = pd.Series(imf, index=train.index[:len(imf)])
                imf_series.name = f"{title} - IMF {i + 1}"

                # Проверка, что IMF содержит числовые данные
                if not np.all(np.isfinite(imf)):
                    print(f"⚠️ IMF {i + 1} содержит нечисловые значения. Пропускаем.")
                    continue

                # Прогнозирование IMF
                forecast_result = base_model_fn(imf_series, f"{title} - IMF {i + 1}", test_size=test_size,
                                                save_plots=False)
                if forecast_result is not None:
                    imf_forecast, metrics = forecast_result
                    if imf_forecast is not None and len(imf_forecast) >= test_size:
                        imf_forecasts.append(imf_forecast.values[:test_size])
                        successful_imfs += 1
                        print(f"✅ IMF {i + 1}: успешно спрогнозировано")
                    else:
                        print(f"⚠️ IMF {i + 1}: не удалось получить прогноз нужной длины")
                else:
                    print(f"⚠️ IMF {i + 1}: прогноз вернул None")
            except Exception as e:
                print(f"❌ Ошибка при прогнозировании IMF {i + 1}: {str(e)}")

        if not imf_forecasts:
            print(f"❌ Не удалось получить прогнозы ни для одного IMF")
            return None, None

        print(f"📊 Успешно спрогнозировано {successful_imfs}/{len(imfs)} IMF компонент")

        # Суммирование прогнозов всех IMF
        min_length = min(len(forecast) for forecast in imf_forecasts)
        combined_forecast = np.sum([forecast[:min_length] for forecast in imf_forecasts], axis=0)

        # Обрезаем или дополняем до размера тестовой выборки
        if len(combined_forecast) < test_size:
            # Дополняем последним значением
            last_value = combined_forecast[-1] if len(combined_forecast) > 0 else np.mean(train.values[-10:])
            padding = np.full(test_size - len(combined_forecast), last_value)
            combined_forecast = np.concatenate([combined_forecast, padding])
        elif len(combined_forecast) > test_size:
            combined_forecast = combined_forecast[:test_size]

        # Расчет метрик
        metrics = calculate_metrics(
            y_true=test.values[:len(combined_forecast)],
            y_pred=combined_forecast,
            y_train=train.values,
            m=infer_period(series)
        )
        metrics['Model'] = model_name

        # Визуализация
        if save_plots:
            setup_plot_style()
            plt.figure(figsize=(12, 6))
            plt.plot(train.index, train.values, 'b-', label='Обучающие данные', linewidth=2)
            plt.plot(test.index[:len(combined_forecast)], test.values[:len(combined_forecast)],
                     'g-', label='Тестовые данные (факт)', linewidth=2)
            plt.plot(test.index[:len(combined_forecast)], combined_forecast,
                     'r--', label=f'{model_name} прогноз (sMAPE={metrics["sMAPE (%)"]:.2f}%)', linewidth=2.5)
            plt.title(f'Прогноз {model_name} для {title}', fontsize=16)
            plt.xlabel('Дата', fontsize=14)
            plt.ylabel('Значение', fontsize=14)
            plt.legend(fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            from config.config import Config
            safe_title = title.replace(" ", "_").replace("+", "_").replace("/", "_")
            safe_model = model_name.replace("+", "_").replace(" ", "_")
            save_path = Config.RESULTS_DIR / f'combined_forecast_{safe_title}_{safe_model}.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"💾 График сохранен: {save_path}")

        print(f"✅ Успешно завершено прогнозирование {model_name} для {title}")
        print(f"📊 Метрики: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, sMAPE={metrics['sMAPE (%)']:.2f}%")

        return pd.Series(combined_forecast, index=test.index[:len(combined_forecast)]), metrics

    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА в комбинированной модели {model_name} для {title}: {str(e)}")
        import traceback
        traceback.print_exc()

        # Фолбэк: наивный прогноз
        test_size_actual = min(test_size, len(series))
        naive_forecast = np.full(test_size_actual, np.median(
            series.iloc[-test_size_actual - 10:-test_size_actual] if len(
                series) > test_size_actual + 10 else series.iloc[-test_size_actual - 1:-1]))

        actual_test_values = series.iloc[-test_size_actual:].values
        actual_train_values = series.iloc[:-test_size_actual].values

        metrics = calculate_metrics(
            y_true=actual_test_values,
            y_pred=naive_forecast,
            y_train=actual_train_values,
            m=infer_period(series)
        )
        metrics['Model'] = f"{model_name}(naive)"
        print("🔄 Используется наивный прогноз как фолбэк")
        print(
            f"📊 Фолбэк метрики: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, sMAPE={metrics['sMAPE (%)']:.2f}%")

        test_index = series.iloc[-test_size_actual:].index
        return pd.Series(naive_forecast, index=test_index), metrics