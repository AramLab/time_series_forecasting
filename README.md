# Time Series Forecasting Project

A comprehensive time series forecasting library featuring multiple advanced models and analysis capabilities with support for parallel processing.

## Features

- **Multiple Forecasting Models**:
  - LSTM (Long Short-Term Memory)
  - ARIMA (AutoRegressive Integrated Moving Average)
  - ETS (Error, Trend, Seasonality)
  - Prophet (Facebook's forecasting tool)
  - CEEMDAN+LSTM (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise + LSTM)
  - CEEMDAN+ARIMA (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise + ARIMA)
  - CEEMDAN+ETS (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise + ETS)
  - CEEMDAN+Prophet (Complete Ensemble Empirical Mode Decomposition with Adaptive Noise + Prophet)

- **Data Processing Modes**:
  - Synthetic time series generation
  - M3/M4 competition datasets
  - Custom dataset loading
  - Configurable number of series and seasonality

- **Modular Architecture**:
  - Clean separation of concerns
  - Easy to extend and maintain
  - Parallel processing capabilities

- **Parallel Processing Support**:
  - Multiprocessing for model evaluation
  - Mac-compatible multiprocessing implementation
  - Configurable number of processes

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from forecasting.models import LSTMModel, ARIMAModel, ETSModel, ProphetModel
from forecasting.ceemdan_models import CEEMDANLSTM, CEEMDANARIMA, CEEMDANETS, CEEMDANProphet
from forecasting.data import DataLoader, SyntheticGenerator

# Load data
data_loader = DataLoader()
data = data_loader.load_m4_data()

# Initialize model
lstm_model = LSTMModel()
predictions = lstm_model.fit_predict(data)

# Or use CEEMDAN ensemble
ceemdan_lstm = CEEMDANLSTM()
predictions = ceemdan_lstm.fit_predict(data)
```

### Running the Full Pipeline

```bash
# Run with synthetic data
python -m forecasting.main --mode synthetic --test-size 12 --max-series 10

# Run with M3 data
python -m forecasting.main --mode m3 --test-size 12 --max-series 10

# Run with M4 data
python -m forecasting.main --mode m4 --test-size 12 --max-series 10

# Run with parallel processing
python -m forecasting.main --mode synthetic --parallel --num-processes 4
```

## Project Structure

```
forecasting/
├── __init__.py
├── main.py                 # Entry point with CLI interface
├── models/
│   ├── __init__.py
│   ├── lstm.py            # LSTM implementation
│   ├── arima.py           # ARIMA implementation
│   ├── ets.py             # ETS implementation
│   └── prophet.py         # Prophet implementation
├── ceemdan_models/
│   ├── __init__.py
│   ├── ceemdan_lstm.py    # CEEMDAN+LSTM implementation
│   ├── ceemdan_arima.py   # CEEMDAN+ARIMA implementation
│   ├── ceemdan_ets.py     # CEEMDAN+ETS implementation
│   └── ceemdan_prophet.py # CEEMDAN+Prophet implementation
├── data/
│   ├── __init__.py
│   ├── loader.py          # Data loading utilities
│   └── synthetic_generator.py # Synthetic data generation
├── utils/
│   ├── __init__.py
│   ├── metrics.py         # Performance metrics
│   └── preprocessing.py   # Data preprocessing
└── config/
    ├── __init__.py
    └── config.py          # Configuration settings
```

## Parallel Processing on Mac

Mac systems use a different multiprocessing start method compared to Linux. The project handles this by:

1. Using `multiprocessing.set_start_method('spawn')` when needed
2. Properly structuring code to avoid issues with pickling
3. Implementing proper process isolation for model training

Example of parallel model evaluation:

```python
from multiprocessing import Pool
import os

def evaluate_model(args):
    model_class, data, params = args
    model = model_class(**params)
    return model.evaluate(data)

if __name__ == "__main__":
    # This is important for Mac compatibility
    if os.name == 'posix':
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    
    # Run evaluations in parallel
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(evaluate_model, model_args_list)
```

## Datasets Support

The project supports various time series datasets:

- M3 Competition Data
- M4 Competition Data
- Custom CSV files
- Synthetic time series generation
- Popular benchmark datasets

## Configuration

All models can be configured with parameters for:
- Seasonality settings
- Number of series to process
- Model hyperparameters
- Training/validation splits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License
