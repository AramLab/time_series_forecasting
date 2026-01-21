"""
Integration Example: Using All New Components

This script demonstrates how to use:
1. IMF Analysis - analyze statistical characteristics of decomposed components
2. CEEMDAN Optimizer - optimize decomposition parameters
3. Complexity Reduction - reduce computational complexity
4. Complexity Paradox Analysis - understand why hybrids fail
5. Adaptive Hybrid Model - adaptive model selection for IMFs
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


def example_imf_analysis():
    """Example 1: Analyzing IMF characteristics"""
    print("\n" + "="*80)
    print("EXAMPLE 1: IMF Analysis")
    print("="*80)
    
    from utils.imf_analysis import (
        IMFAnalyzer, 
        analyze_imf_components, 
        get_model_recommendation_for_imf,
        compute_imf_weights
    )
    
    # Create synthetic IMF-like components
    t = np.linspace(0, 10, 500)
    
    # Trend-like IMF (low frequency, smooth)
    trend_imf = 0.1 * t + 0.5 * np.sin(0.5 * t)
    
    # Seasonal-like IMF (periodic)
    seasonal_imf = 2.0 * np.sin(2 * np.pi * t / 3)
    
    # Noise-like IMF (random)
    noise_imf = 0.5 * np.random.randn(500)
    
    imfs = [trend_imf, seasonal_imf, noise_imf]
    
    # Analyze components
    print("\n📊 Analyzing IMF components:")
    analyzers = analyze_imf_components(imfs, verbose=True)
    
    # Get model recommendations
    print("\n🤖 Model recommendations for each IMF:")
    for i, imf in enumerate(imfs):
        model_scores = get_model_recommendation_for_imf(imf, detailed=True)
        best_model = max(model_scores.items(), key=lambda x: x[1])[0]
        print(f"\n  IMF {i+1}:")
        for model, score in model_scores.items():
            marker = "✓" if model == best_model else " "
            print(f"    {marker} {model}: {score:.4f}")
    
    # Compute adaptive weights
    print("\n⚖️ Adaptive weights:")
    weights_energy = compute_imf_weights(imfs, method='energy')
    weights_entropy = compute_imf_weights(imfs, method='entropy')
    
    print(f"  By energy:  {weights_energy}")
    print(f"  By entropy: {weights_entropy}")
    
    return analyzers


def example_ceemdan_optimization():
    """Example 2: Optimizing CEEMDAN parameters"""
    print("\n" + "="*80)
    print("EXAMPLE 2: CEEMDAN Parameter Optimization")
    print("="*80)
    
    from optimization.ceemdan_optimizer import CEEMDANOptimizer
    from models.ceemdan_models import safe_import_ceemdan
    
    # Load or create test series
    np.random.seed(42)
    t = np.linspace(0, 20, 200)
    series = 10 + 2*np.sin(0.5*t) + 0.5*np.random.randn(200)
    
    CEEMDAN_Class = safe_import_ceemdan()
    if CEEMDAN_Class is None:
        print("❌ CEEMDAN not available for optimization")
        return None
    
    # Create decomposition function
    def decompose_func(data, trials, noise_width):
        ceemdan = CEEMDAN_Class(trials=trials, noise_width=noise_width)
        return ceemdan(data.astype(float))
    
    # Optimize parameters
    print("\n🔍 Optimizing CEEMDAN parameters with grid search...")
    optimizer = CEEMDANOptimizer(decompose_func, series)
    
    results = optimizer.grid_search(
        trials_range=(10, 50, 10),
        noise_width_range=(0.01, 0.10, 0.02)
    )
    
    print(f"\n📋 Results:")
    print(f"  Best parameters: {results['best_params']}")
    print(f"  Best score: {results['best_score']:.4f}")
    
    if 'all_results' in results:
        print(f"  Total configurations tested: {len(results['all_results'])}")
    
    return results


def example_complexity_reduction():
    """Example 3: Reducing computational complexity"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Complexity Reduction Techniques")
    print("="*80)
    
    from optimization.complexity_reduction import (
        ComplexityReducer,
        AdaptiveHybridForecaster,
        estimate_forecasting_complexity
    )
    
    # Create synthetic IMFs
    np.random.seed(42)
    t = np.linspace(0, 10, 500)
    imfs = [
        np.sin(k * t) + 0.1 * np.random.randn(500)
        for k in np.linspace(0.5, 5, 10)
    ]
    
    print("\n🔧 Complexity Reduction Operations:")
    
    # 1. IMF Pruning
    print("\n1️⃣ IMF Pruning (remove low-energy components):")
    pruned_imfs, prune_info = ComplexityReducer.prune_imfs(imfs, energy_threshold=0.01)
    print(f"   Original IMFs: {prune_info['original_count']}")
    print(f"   Pruned IMFs: {prune_info['pruned_count']}")
    print(f"   Removed: {len(prune_info['removed_indices'])} components")
    print(f"   Energy retained: {prune_info['energy_retained']:.2%}")
    
    # 2. Adaptive trials estimation
    print("\n2️⃣ Adaptive CEEMDAN trials:")
    series = np.concatenate(imfs)
    recommended_trials, trials_info = ComplexityReducer.adaptive_ceemdan_trials(series)
    print(f"   Series complexity: {trials_info['complexity_level']}")
    print(f"   Complexity indicator: {trials_info['complexity_indicator']:.3f}")
    print(f"   Recommended trials: {recommended_trials}")
    
    # 3. Complexity estimation
    print("\n3️⃣ Forecasting complexity estimation:")
    complexity = estimate_forecasting_complexity(imfs, n_models=4, parallel_jobs=4)
    print(f"   Total IMF forecasts: {complexity['total_imf_forecasts']}")
    print(f"   Serial time estimate: {complexity['estimated_time_serial']:.2f}")
    print(f"   Parallel time estimate: {complexity['estimated_time_parallel']:.2f}")
    print(f"   Speedup: {complexity['speedup_serial_to_parallel']:.2f}×")
    print(f"   With pruning: {complexity['speedup_with_pruning']:.2f}×")
    
    # 4. Adaptive forecaster
    print("\n4️⃣ Adaptive hybrid forecaster:")
    forecaster = AdaptiveHybridForecaster(n_jobs=1, prune_threshold=0.01)
    processed_imfs, metadata = forecaster.preprocess_imfs(imfs, verbose=True)


def example_complexity_paradox_analysis():
    """Example 4: Analyzing the complexity paradox"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Complexity Paradox Analysis")
    print("="*80)
    
    from analysis.complexity_paradox import ComplexityParadoxAnalyzer
    
    analyzer = ComplexityParadoxAnalyzer()
    
    # Show theoretical explanation
    print("\n📖 Theoretical Explanation of Complexity Paradox:")
    print("-" * 80)
    explanation = analyzer.theoretical_explanation()
    # Print first 2000 characters
    print(explanation[:2000] + "\n...[see full documentation]...\n")
    
    # Demonstrate error accumulation analysis
    print("\n📊 Error Accumulation Analysis:")
    print("-" * 80)
    
    # Simulate forecast errors for 5 IMF components
    np.random.seed(42)
    imf_errors = [
        np.random.randn(100) * (0.05 * (i+1))  # Increasing error magnitude
        for i in range(5)
    ]
    
    accumulation = analyzer.forecast_error_accumulation_analysis(imf_errors)
    
    print(f"Number of IMFs: {accumulation['n_imfs']}")
    print(f"Individual RMSEs: {accumulation['individual_rmse']}")
    print(f"Combined RMSE: {accumulation['combined_rmse']:.4f}")
    print(f"RMS of individual RMSEs: {accumulation['rms_of_individual_rmse']:.4f}")
    print(f"\nError correlation indicator: {accumulation['error_correlation_indicator']:.4f}")
    print(f"  (negative = better error cancellation, positive = error amplification)")


def example_adaptive_model_selection():
    """Example 5: Using adaptive model selection"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Adaptive Model Selection in Action")
    print("="*80)
    
    from utils.imf_analysis import IMFAnalyzer
    
    # Create different types of IMF-like components
    t = np.linspace(0, 10, 300)
    
    components = {
        'Trend': 0.1 * t + np.sin(0.2 * t),
        'Seasonal': 2.0 * np.sin(2 * np.pi * t / 3),
        'Noise': 0.5 * np.random.randn(300),
        'Mixed': 0.1 * t + 2.0 * np.sin(2 * np.pi * t / 3) + 0.3 * np.random.randn(300)
    }
    
    print("\n🤖 Adaptive model selection results:")
    print("-" * 80)
    
    for name, component in components.items():
        analyzer = IMFAnalyzer(component)
        classification = analyzer.classify()
        
        # Get detailed model scores
        from utils.imf_analysis import get_model_recommendation_for_imf
        scores = get_model_recommendation_for_imf(component, detailed=True)
        
        best_model = max(scores.items(), key=lambda x: x[1])[0]
        
        chars = analyzer.get_characteristics()
        
        print(f"\n{name} Component:")
        print(f"  Classification: {classification}")
        print(f"  Entropy: {chars['entropy']:.3f}")
        print(f"  Frequency concentration: {chars['freq_concentration']:.3f}")
        print(f"  Selected model: {best_model} ✓")
        print(f"  All scores: {', '.join([f'{m}={s:.3f}' for m, s in scores.items()])}")


def example_comprehensive_workflow():
    """Example 6: Comprehensive workflow using all components"""
    print("\n" + "="*80)
    print("EXAMPLE 6: Comprehensive Hybrid Forecasting Workflow")
    print("="*80)
    
    print("""
This example shows the complete workflow:
1. Load data
2. Perform CEEMDAN decomposition (with optimized parameters)
3. Analyze IMF components
4. Select optimal model for each component
5. Apply complexity reduction
6. Forecast and analyze results
7. Compare to simple baseline

Note: Full implementation would require actual model functions
(ARIMA, ETS, LSTM, Prophet) and a complete time series.

For a working example, see analysis/synthetic_data_analysis.py
which implements the complete workflow on synthetic data.
    """)


def main():
    """Run all examples"""
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + " Advanced Time Series Forecasting Components - Integration Examples ".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        # Example 1: IMF Analysis
        example_imf_analysis()
        
        # Example 2: CEEMDAN Optimization
        example_ceemdan_optimization()
        
        # Example 3: Complexity Reduction
        example_complexity_reduction()
        
        # Example 4: Complexity Paradox Analysis
        example_complexity_paradox_analysis()
        
        # Example 5: Adaptive Model Selection
        example_adaptive_model_selection()
        
        # Example 6: Comprehensive Workflow
        example_comprehensive_workflow()
        
        print("\n" + "="*80)
        print("✅ All examples completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during examples: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
