"""
Main entry point for the Time Series Forecasting Project
Demonstrates all models and parallel processing capabilities
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import multiprocessing as mp
import os
import argparse
from functools import partial

# Import all models
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import LSTMModel, ARIMAModel, ETSModel, ProphetModel
from ceemdan_models import CEEMDANLSTM, CEEMDANARIMA, CEEMDANETS, CEEMDANProphet
from data import DataLoader, SyntheticGenerator
from utils.metrics import calculate_metrics
from config.config import Config


def evaluate_model_on_series(model_info: tuple, series_data: np.ndarray, steps: int = 12):
    """
    Evaluate a model on a single time series
    
    Args:
        model_info: Tuple containing (model_class, model_name, model_params)
        series_data: Time series data to evaluate on
        steps: Number of steps to forecast
        
    Returns:
        Dictionary with evaluation results
    """
    model_class, model_name, model_params = model_info
    
    try:
        # Create and train model
        model = model_class(**model_params)
        
        # Split data into train and test
        train_data = series_data[:-steps]
        actual_test = series_data[-steps:]
        
        if len(train_data) < 10:  # Need minimum data for training
            return {
                'model_name': model_name,
                'series_id': id(series_data),
                'error': 'Insufficient training data'
            }
        
        # Fit and predict
        predictions = model.fit_predict(train_data, steps)
        
        # Calculate metrics
        metrics = calculate_metrics(actual_test, predictions)
        
        return {
            'model_name': model_name,
            'series_id': id(series_data),
            'predictions': predictions,
            'actual': actual_test,
            'metrics': metrics
        }
    except Exception as e:
        return {
            'model_name': model_name,
            'series_id': id(series_data),
            'error': str(e)
        }


def run_single_series_evaluation(series_data: np.ndarray, 
                                models_config: List[tuple], 
                                steps: int = 12):
    """
    Evaluate all models on a single time series
    
    Args:
        series_data: Time series data to evaluate on
        models_config: List of model configurations
        steps: Number of steps to forecast
        
    Returns:
        List of evaluation results for all models
    """
    results = []
    for model_info in models_config:
        result = evaluate_model_on_series(model_info, series_data, steps)
        results.append(result)
    return results


def run_parallel_evaluation(all_series: List[np.ndarray], 
                          models_config: List[tuple], 
                          steps: int = 12,
                          num_processes: int = None):
    """
    Run parallel evaluation of models on multiple time series
    
    Args:
        all_series: List of time series to evaluate
        models_config: List of model configurations
        steps: Number of steps to forecast
        num_processes: Number of processes to use (default: CPU count)
        
    Returns:
        List of evaluation results
    """
    if num_processes is None:
        num_processes = min(mp.cpu_count(), len(all_series))
    
    print(f"Running parallel evaluation with {num_processes} processes on {len(all_series)} series...")
    
    # Create partial function with fixed parameters
    eval_func = partial(run_single_series_evaluation, models_config=models_config, steps=steps)
    
    # Use multiprocessing pool
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(eval_func, all_series)
    
    # Flatten results
    flattened_results = []
    for series_results in results:
        flattened_results.extend(series_results)
    
    return flattened_results


def run_serial_evaluation(all_series: List[np.ndarray], 
                        models_config: List[tuple], 
                        steps: int = 12):
    """
    Run serial evaluation of models on multiple time series
    
    Args:
        all_series: List of time series to evaluate
        models_config: List of model configurations
        steps: Number of steps to forecast
        
    Returns:
        List of evaluation results
    """
    print(f"Running serial evaluation on {len(all_series)} series...")
    
    results = []
    for i, series_data in enumerate(all_series):
        print(f"Evaluating series {i+1}/{len(all_series)}")
        series_results = run_single_series_evaluation(series_data, models_config, steps)
        results.extend(series_results)
    
    return results


def setup_models_config(config_manager=None):
    """
    Setup configuration for all models
    
    Args:
        config_manager: Config object with default parameters
        
    Returns:
        List of model configurations
    """
    if config_manager is None:
        config_manager = Config()
    
    models_config = [
        (LSTMModel, 'LSTM', config_manager.get_model_config('lstm')),
        (ARIMAModel, 'ARIMA', config_manager.get_model_config('arima')),
        (ETSModel, 'ETS', config_manager.get_model_config('ets')),
        (ProphetModel, 'Prophet', config_manager.get_model_config('prophet')),
        (CEEMDANLSTM, 'CEEMDAN+LSTM', config_manager.get_model_config('ceemdan_lstm')),
        (CEEMDANARIMA, 'CEEMDAN+ARIMA', config_manager.get_model_config('ceemdan_arima')),
        (CEEMDANETS, 'CEEMDAN+ETS', config_manager.get_model_config('ceemdan_ets')),
        (CEEMDANProphet, 'CEEMDAN+Prophet', config_manager.get_model_config('ceemdan_prophet'))
    ]
    
    return models_config


def main():
    """
    Main function to demonstrate the time series forecasting project
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Time Series Forecasting Project')
    parser.add_argument('--mode', choices=['synthetic', 'm3', 'm4', 'custom'], 
                       default='synthetic', help='Data source mode')
    parser.add_argument('--test-size', type=int, default=12, 
                       help='Number of test samples (default: 12)')
    parser.add_argument('--max-series', type=int, default=5, 
                       help='Maximum number of series to process (default: 5)')
    parser.add_argument('--parallel', action='store_true', 
                       help='Use parallel processing')
    parser.add_argument('--num-processes', type=int, 
                       help='Number of processes to use (default: CPU count)')
    parser.add_argument('--seasonal-period', type=int, default=12, 
                       help='Seasonal period for metrics calculation')
    
    args = parser.parse_args()
    
    # Handle multiprocessing on Mac
    if os.name == 'posix':  # Mac/Linux
        mp.set_start_method('spawn', force=True)
    
    print("Time Series Forecasting Project")
    print("="*40)
    print(f"Mode: {args.mode}")
    print(f"Test size: {args.test_size}")
    print(f"Max series: {args.max_series}")
    print(f"Parallel: {args.parallel}")
    print(f"Seasonal period: {args.seasonal_period}")
    print()
    
    # Initialize configuration
    config_manager = Config()
    
    # Load or generate data
    if args.mode == 'synthetic':
        print("Generating synthetic data...")
        generator = SyntheticGenerator()
        all_series = generator.generate_multiple_series(num_series=args.max_series)
    else:
        print(f"Loading {args.mode.upper()} data...")
        loader = DataLoader()
        all_series = loader.load_dataset(args.mode, limit=args.max_series)
    
    print(f"Loaded {len(all_series)} time series")
    for i, series in enumerate(all_series):
        print(f"Series {i+1}: length={len(series)}")
    print()
    
    # Setup model configurations
    models_config = setup_models_config(config_manager)
    print(f"Initialized {len(models_config)} models:")
    for _, name, _ in models_config:
        print(f"  - {name}")
    print()
    
    # Run evaluation
    if args.parallel:
        results = run_parallel_evaluation(
            all_series, 
            models_config, 
            steps=args.test_size,
            num_processes=args.num_processes
        )
    else:
        results = run_serial_evaluation(
            all_series, 
            models_config, 
            steps=args.test_size
        )
    
    # Process and display results
    print("\nEvaluation Results:")
    print("="*40)
    
    # Group results by model
    model_results = {}
    for result in results:
        model_name = result.get('model_name', 'Unknown')
        if model_name not in model_results:
            model_results[model_name] = []
        if 'error' not in result:
            model_results[model_name].append(result)
    
    # Calculate average metrics for each model
    for model_name, model_results_list in model_results.items():
        if model_results_list:
            avg_rmse = np.mean([r['metrics']['rmse'] for r in model_results_list])
            avg_mae = np.mean([r['metrics']['mae'] for r in model_results_list])
            avg_smape = np.mean([r['metrics']['smape'] for r in model_results_list])
            avg_mase = np.mean([r['metrics']['mase'] for r in model_results_list])
            
            print(f"{model_name}:")
            print(f"  RMSE: {avg_rmse:.4f}")
            print(f"  MAE: {avg_mae:.4f}")
            print(f"  SMAPE: {avg_smape:.4f}%")
            print(f"  MASE: {avg_mase:.4f}")
            print()
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()