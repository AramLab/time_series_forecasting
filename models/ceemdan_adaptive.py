"""
Advanced CEEMDAN Hybrid Models with Adaptive Model Selection

Implements improved hybrid forecasting with:
1. Adaptive model selection based on IMF characteristics
2. Optimized CEEMDAN parameters
3. Complexity reduction techniques
4. Weighted IMF ensemble
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def ceemdan_adaptive_hybrid_model(
    series, 
    model_functions, 
    title, 
    test_size=24, 
    model_name="CEEMDAN+Adaptive",
    save_plots=True,
    use_complexity_reduction=True,
    adaptive_params=None
):
    """
    Advanced CEEMDAN hybrid model with adaptive model selection.
    
    Parameters:
    -----------
    series : pd.Series
        Input time series
    model_functions : dict
        Dictionary of model functions {model_name: forecast_function}
    title : str
        Title for the series
    test_size : int
        Size of test set
    model_name : str
        Name of the model
    save_plots : bool
        Whether to save plots
    use_complexity_reduction : bool
        Whether to apply complexity reduction techniques
    adaptive_params : dict
        Custom parameters for adaptation
        
    Returns:
    --------
    tuple : (forecast_series, metrics_dict)
    """
    try:
        from utils.visualization import setup_plot_style
        from utils.metrics import calculate_metrics
        from utils.preprocessing import infer_period
        from utils.imf_analysis import analyze_imf_components, get_model_recommendation_for_imf, compute_imf_weights
        from optimization.complexity_reduction import AdaptiveHybridForecaster, estimate_forecasting_complexity
        from models.ceemdan_models import safe_import_ceemdan
        
        # Split data
        train = series.iloc[:-test_size]
        test = series.iloc[-test_size:]
        
        # Get CEEMDAN class
        CEEMDAN_Class = safe_import_ceemdan()
        if CEEMDAN_Class is None:
            print(f"❌ CEEMDAN not available")
            return None, None
        
        # Initialize complexity reducer if needed
        forecaster = None
        if use_complexity_reduction:
            forecaster = AdaptiveHybridForecaster(n_jobs=1, prune_threshold=0.01)
            recommended_trials, trials_info = forecaster.estimate_optimal_trials(train.values, verbose=True)
        else:
            recommended_trials = 20
        
        print(f"🔍 CEEMDAN decomposition with trials={recommended_trials}...")
        
        # Decompose
        ceemdan_instance = CEEMDAN_Class(trials=recommended_trials, noise_width=0.05)
        imfs = ceemdan_instance(train.values.astype(float))
        print(f"✅ Obtained {len(imfs)} IMF components")
        
        # Analyze IMF components
        imf_analyzers = analyze_imf_components(imfs, verbose=True)
        
        # Apply complexity reduction if enabled
        if use_complexity_reduction and forecaster is not None:
            imfs_processed, preprocess_info = forecaster.preprocess_imfs(imfs, verbose=True)
            # Update analyzers for processed IMFs
            imf_analyzers = analyze_imf_components(imfs_processed, verbose=False)
        else:
            imfs_processed = imfs
        
        # Adaptive model selection and forecasting
        print(f"\n📊 Adaptive model selection for {len(imfs_processed)} components:")
        imf_forecasts = []
        selected_models = []
        imf_weights = []
        
        for i, imf in enumerate(imfs_processed):
            try:
                # Get model recommendation for this IMF
                model_scores = get_model_recommendation_for_imf(imf, detailed=True)
                best_model = max(model_scores.items(), key=lambda x: x[1])[0]
                
                # Check if model function is available
                if best_model not in model_functions:
                    # Fallback to first available model
                    best_model = list(model_functions.keys())[0]
                
                print(f"  IMF {i+1}: {best_model} (scores: {', '.join([f'{m}={s:.3f}' for m, s in model_scores.items()])})")
                
                # Create IMF series
                imf_series = pd.Series(imf, index=train.index[:len(imf)])
                imf_series.name = f"{title} - IMF {i+1}"
                
                # Forecast
                forecast_result = model_functions[best_model](imf_series, imf_series.name, test_size=test_size)
                
                if forecast_result is not None:
                    imf_forecast, _ = forecast_result
                    if imf_forecast is not None and len(imf_forecast) >= test_size:
                        imf_forecasts.append(imf_forecast.values[:test_size])
                        selected_models.append(best_model)
                        
                        # Compute weight based on IMF energy
                        imf_weight = np.sum(imf ** 2)
                        imf_weights.append(imf_weight)
                        
                        print(f"    ✅ Forecast successful, weight={imf_weight:.4e}")
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
                continue
        
        if not imf_forecasts:
            print(f"❌ No successful forecasts obtained")
            return None, None
        
        # Normalize weights
        imf_weights = np.array(imf_weights)
        imf_weights = imf_weights / np.sum(imf_weights)
        
        # Combine forecasts with adaptive weights
        min_length = min(len(forecast) for forecast in imf_forecasts)
        combined_forecast = np.zeros(min_length)
        
        for forecast, weight in zip(imf_forecasts, imf_weights):
            combined_forecast += forecast[:min_length] * weight
        
        # Pad/trim to test_size
        if len(combined_forecast) < test_size:
            last_value = combined_forecast[-1] if len(combined_forecast) > 0 else np.mean(train.values[-10:])
            padding = np.full(test_size - len(combined_forecast), last_value)
            combined_forecast = np.concatenate([combined_forecast, padding])
        elif len(combined_forecast) > test_size:
            combined_forecast = combined_forecast[:test_size]
        
        # Calculate metrics
        metrics = calculate_metrics(
            y_true=test.values[:len(combined_forecast)],
            y_pred=combined_forecast,
            y_train=train.values,
            m=infer_period(series)
        )
        metrics['Model'] = model_name
        metrics['N_IMFs'] = len(imf_forecasts)
        metrics['Selected_Models'] = ', '.join(selected_models)
        
        # Visualization
        if save_plots:
            setup_plot_style()
            plt.figure(figsize=(14, 7))
            
            plt.plot(train.index, train.values, 'b-', label='Training data', linewidth=2)
            plt.plot(test.index[:len(combined_forecast)], test.values[:len(combined_forecast)],
                    'g-', label='Actual test data', linewidth=2)
            plt.plot(test.index[:len(combined_forecast)], combined_forecast,
                    'r--', label=f'{model_name} forecast (sMAPE={metrics["sMAPE (%)"]:.2f}%)', linewidth=2.5)
            
            plt.title(f'{model_name} Forecast for {title}', fontsize=14, fontweight='bold')
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Value', fontsize=12)
            plt.legend(fontsize=11, loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            from config.config import Config
            safe_title = title.replace(" ", "_").replace("+", "_").replace("/", "_")
            safe_model = model_name.replace("+", "_").replace(" ", "_")
            save_path = Config.RESULTS_DIR / f'adaptive_forecast_{safe_title}_{safe_model}.png'
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"💾 Plot saved: {save_path}")
        
        print(f"✅ {model_name} completed for {title}")
        print(f"📊 Metrics: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, sMAPE={metrics['sMAPE (%)']:.2f}%")
        print(f"📋 Used {len(imf_forecasts)} IMFs with models: {', '.join(set(selected_models))}")
        
        return pd.Series(combined_forecast, index=test.index[:len(combined_forecast)]), metrics
    
    except Exception as e:
        print(f"❌ CRITICAL ERROR in {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Fallback to naive forecast
        from utils.metrics import calculate_metrics
        from utils.preprocessing import infer_period
        
        test_size_actual = min(test_size, len(series))
        naive_forecast = np.full(test_size_actual, np.median(series.iloc[-test_size_actual-10:-test_size_actual]))
        
        metrics = calculate_metrics(
            y_true=series.iloc[-test_size_actual:].values,
            y_pred=naive_forecast,
            y_train=series.iloc[:-test_size_actual].values,
            m=infer_period(series)
        )
        metrics['Model'] = f"{model_name}(naive_fallback)"
        
        return pd.Series(naive_forecast, index=series.iloc[-test_size_actual:].index), metrics
