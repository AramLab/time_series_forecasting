# Time Series Forecasting Platform | Платформа прогнозирования временных рядов

[English](#english) | [Русский](#русский)

---

## English

### Overview

A comprehensive time series forecasting platform that combines multiple forecasting models with advanced decomposition techniques. The system supports ARIMA, ETS (Error-Trend-Seasonality), Prophet, LSTM, and CEEMDAN-based ensemble methods for accurate predictions on various time series datasets.

### Features

- **Multiple Forecasting Models**
  - ARIMA (AutoRegressive Integrated Moving Average)
  - ETS (Error-Trend-Seasonality)
  - Prophet (Facebook's forecasting tool)
  - LSTM (Deep Learning)
  - CEEMDAN+ARIMA (Ensemble method)
  - CEEMDAN+ETS (Ensemble method)

- **Advanced Decomposition**
  - CEEMDAN (Complete Ensemble EMD with Adaptive Noise)
  - Automatic IMF (Intrinsic Mode Function) extraction
  - Graceful fallback to pure Python implementation

- **Multiple Data Modes**
  - Synthetic data generation
  - M3 competition datasets
  - M4 competition datasets
  - Custom CSV files

- **Comprehensive Analysis**
  - Automatic hyperparameter optimization
  - Performance metrics (RMSE, MAE, sMAPE, MASE)
  - Comparative model analysis
  - Visualization of forecasts and components

### Quick Start

#### Local Installation

```bash
# Clone repository
git clone https://github.com/AramLab/time_series_forecasting.git
cd time_series_forecasting

# Install dependencies
pip install -r requirements.txt

# Run with synthetic data
python main.py --mode synthetic --test-size 12

# Run with M3 competition data
python main.py --mode m3 --max-rows 10

# Run with custom CSV
python main.py --mode csv --file data.csv --test-size 12
```

#### Docker Installation

```bash
# Build Docker image
docker-compose build

# Run in Docker
docker-compose run forecasting python main.py --mode synthetic --test-size 12

# Or use docker-compose directly
docker-compose up
```

### Command Line Options

```
python main.py [OPTIONS]

Options:
  --mode {synthetic,m3,m4,csv}    Data source mode (default: synthetic)
  --test-size INT                 Number of test samples (default: 12)
  --file PATH                     Path to CSV file (required for --mode csv)
  --max-rows INT                  Maximum rows to process from M3/M4 (default: 10)
  --verbose                       Enable verbose output
```

### Project Structure

```
time_series_forecasting/
├── main.py                       # Entry point
├── config/
│   └── config.py                # Configuration settings
├── data/
│   ├── data_loader.py           # Data loading utilities
│   └── __init__.py
├── models/
│   ├── arima_model.py           # ARIMA implementation
│   ├── ets_model.py             # ETS implementation
│   ├── prophet_model.py         # Prophet wrapper
│   ├── lstm_model.py            # LSTM implementation
│   ├── ceemdan_models.py        # CEEMDAN wrapper
│   └── __init__.py
├── utils/
│   ├── ceemdan_pure_python.py   # Pure Python CEEMDAN fallback
│   ├── metrics.py               # Performance metrics
│   ├── preprocessing.py         # Data preprocessing
│   ├── visualization.py         # Plotting utilities
│   └── __init__.py
├── optimization/
│   ├── ceemdan_optimizer.py     # CEEMDAN optimization
│   ├── complexity_reduction.py  # Complexity analysis
│   └── __init__.py
├── analysis/
│   ├── complexity_paradox.py    # Complexity paradox analysis
│   ├── m3_m4_analysis.py        # Competition data analysis
│   └── __init__.py
└── results/                      # Output directory for plots and CSV
```

### Model Comparison

The platform supports comprehensive model comparison with metrics:

| Model | RMSE | MAE | sMAPE (%) | MASE |
|-------|------|-----|-----------|------|
| ARIMA | Lower | Lower | 2-3% | Good |
| ETS | Lower | Lower | 2-3% | Good |
| Prophet | Moderate | Moderate | 2-4% | Moderate |
| LSTM | Higher | Higher | 5-6% | Poor |
| CEEMDAN+ARIMA | Higher | Higher | 4-5% | Moderate |
| CEEMDAN+ETS | Moderate | Moderate | 2-3% | Good |

*Metrics vary based on dataset characteristics*

### CEEMDAN and PyEMD

The platform uses CEEMDAN for time series decomposition with two implementations:

1. **PyEMD (Preferred)** - C-extension based, fast but requires compilation
2. **SimpleCEEMDAN (Fallback)** - Pure Python, slower but always works

When PyEMD is unavailable (common on Mac/ARM64), the system automatically falls back to SimpleCEEMDAN with a warning message. This is **normal and expected behavior**.

### Output

- **Plots**: Individual model forecasts, decomposition visualizations
- **CSV**: Comprehensive results with metrics for all models
- **Console**: Real-time progress and model performance

### Requirements

- Python 3.10+
- NumPy, Pandas, SciPy
- Scikit-learn, Statsmodels
- Prophet, TensorFlow/Keras
- PyEMD (optional, with pure Python fallback)

### Docker

The project includes optimized Dockerfile with:
- Multi-stage build for smaller images
- Python 3.10 for better wheel compatibility
- All scientific dependencies pre-compiled

### License

MIT License - see LICENSE file for details

### Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

## Русский

### Описание

Комплексная платформа для прогнозирования временных рядов, которая объединяет несколько моделей прогнозирования с передовыми методами декомпозиции. Система поддерживает ARIMA, ETS (Error-Trend-Seasonality), Prophet, LSTM и ансамбльные методы на основе CEEMDAN для точного прогнозирования различных временных рядов.

### Возможности

- **Несколько моделей прогнозирования**
  - ARIMA (AutoRegressive Integrated Moving Average)
  - ETS (Error-Trend-Seasonality)
  - Prophet (инструмент прогнозирования от Facebook)
  - LSTM (глубокое обучение)
  - CEEMDAN+ARIMA (Ансамбльный метод)
  - CEEMDAN+ETS (Ансамбльный метод)

- **Продвинутая декомпозиция**
  - CEEMDAN (Complete Ensemble EMD with Adaptive Noise)
  - Автоматическое извлечение IMF (Intrinsic Mode Functions)
  - Корректное отступление на чистую реализацию на Python

- **Множество режимов данных**
  - Генерация синтетических данных
  - Датасеты конкурса M3
  - Датасеты конкурса M4
  - Пользовательские CSV файлы

- **Комплексный анализ**
  - Автоматическая оптимизация гиперпараметров
  - Метрики производительности (RMSE, MAE, sMAPE, MASE)
  - Сравнительный анализ моделей
  - Визуализация прогнозов и компонент

### Быстрый старт

#### Локальная установка

```bash
# Клонировать репозиторий
git clone https://github.com/AramLab/time_series_forecasting.git
cd time_series_forecasting

# Установить зависимости
pip install -r requirements.txt

# Запустить с синтетическими данными
python main.py --mode synthetic --test-size 12

# Запустить с данными конкурса M3
python main.py --mode m3 --max-rows 10

# Запустить с пользовательским CSV
python main.py --mode csv --file data.csv --test-size 12
```

#### Установка через Docker

```bash
# Собрать Docker образ
docker-compose build

# Запустить в Docker
docker-compose run forecasting python main.py --mode synthetic --test-size 12

# Или использовать docker-compose напрямую
docker-compose up
```

### Параметры командной строки

```
python main.py [OPTIONS]

Параметры:
  --mode {synthetic,m3,m4,csv}    Источник данных (по умолчанию: synthetic)
  --test-size INT                 Количество тестовых образцов (по умолчанию: 12)
  --file PATH                     Путь к CSV файлу (требуется для --mode csv)
  --max-rows INT                  Макс. строк из M3/M4 (по умолчанию: 10)
  --verbose                       Подробный вывод
```

### Структура проекта

```
time_series_forecasting/
├── main.py                       # Точка входа
├── config/
│   └── config.py                # Настройки конфигурации
├── data/
│   ├── data_loader.py           # Утилиты загрузки данных
│   └── __init__.py
├── models/
│   ├── arima_model.py           # Реализация ARIMA
│   ├── ets_model.py             # Реализация ETS
│   ├── prophet_model.py         # Обёртка Prophet
│   ├── lstm_model.py            # Реализация LSTM
│   ├── ceemdan_models.py        # Обёртка CEEMDAN
│   └── __init__.py
├── utils/
│   ├── ceemdan_pure_python.py   # CEEMDAN на чистом Python
│   ├── metrics.py               # Метрики производительности
│   ├── preprocessing.py         # Предварительная обработка
│   ├── visualization.py         # Утилиты построения графиков
│   └── __init__.py
├── optimization/
│   ├── ceemdan_optimizer.py     # Оптимизация CEEMDAN
│   ├── complexity_reduction.py  # Анализ сложности
│   └── __init__.py
├── analysis/
│   ├── complexity_paradox.py    # Анализ парадокса сложности
│   ├── m3_m4_analysis.py        # Анализ данных конкурса
│   └── __init__.py
└── results/                      # Директория вывода (графики и CSV)
```

### Сравнение моделей

Платформа поддерживает комплексное сравнение моделей с метриками:

| Модель | RMSE | MAE | sMAPE (%) | MASE |
|--------|------|-----|-----------|------|
| ARIMA | Ниже | Ниже | 2-3% | Хорошо |
| ETS | Ниже | Ниже | 2-3% | Хорошо |
| Prophet | Среднее | Среднее | 2-4% | Среднее |
| LSTM | Выше | Выше | 5-6% | Плохо |
| CEEMDAN+ARIMA | Выше | Выше | 4-5% | Среднее |
| CEEMDAN+ETS | Среднее | Среднее | 2-3% | Хорошо |

*Метрики варьируются в зависимости от характеристик датасета*

### CEEMDAN и PyEMD

Платформа использует CEEMDAN для декомпозиции временных рядов с двумя реализациями:

1. **PyEMD (Предпочтительнее)** - на основе C-расширения, быстро, но требует компиляции
2. **SimpleCEEMDAN (Fallback)** - на чистом Python, медленнее, но всегда работает

Когда PyEMD недоступен (часто на Mac/ARM64), система автоматически переходит на SimpleCEEMDAN с предупреждающим сообщением. Это **нормальное и ожидаемое поведение**.

### Вывод

- **Графики**: Прогнозы отдельных моделей, визуализация декомпозиции
- **CSV**: Комплексные результаты с метриками для всех моделей
- **Консоль**: Прогресс в реальном времени и производительность моделей

### Требования

- Python 3.10+
- NumPy, Pandas, SciPy
- Scikit-learn, Statsmodels
- Prophet, TensorFlow/Keras
- PyEMD (опционально, с fallback на чистый Python)

### Docker

Проект включает оптимизированный Dockerfile с:
- Многоэтапной сборкой для меньших образов
- Python 3.10 для лучшей совместимости wheel'ов
- Всеми научными зависимостями предварительно скомпилированными

### Лицензия

MIT License - см. файл LICENSE для подробностей

### Участие в разработке

Приветствуются исправления ошибок и улучшения! Не стесняйтесь создавать issues или pull requests.

---

**Last Updated**: January 22, 2026
