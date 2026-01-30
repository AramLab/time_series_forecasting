# Parallel Processing on Mac Systems

This document explains how to leverage the parallel processing capabilities of the Time Series Forecasting Project on Mac systems.

## Why Mac Requires Special Handling

Mac systems use a different multiprocessing start method compared to Linux systems. Specifically:

- **Linux**: Uses `fork` by default, which copies the parent process
- **Mac**: Uses `spawn` by default, which starts fresh Python processes

This difference can cause issues when sharing objects between processes, especially complex model objects.

## How Our Project Handles Mac Compatibility

The project automatically handles Mac multiprocessing through these mechanisms:

### 1. Process Isolation

Each model runs in its own isolated process space:

```python
import multiprocessing as mp
import os

def evaluate_model(args):
    """
    Evaluate a model on a single time series
    Each process gets its own copy of the model
    """
    model_class, model_params, series_data, steps = args
    
    # Each process creates its own model instance
    model = model_class(**model_params)
    
    # Split data for training and testing
    train_data = series_data[:-steps]
    actual_test = series_data[-steps:]
    
    # Train and predict
    predictions = model.fit_predict(train_data, steps)
    
    return predictions, actual_test

if __name__ == "__main__":
    # Important: This ensures compatibility on Mac
    if os.name == 'posix':  # Mac/Linux
        mp.set_start_method('spawn', force=True)
    
    # Use multiprocessing pool
    with mp.Pool() as pool:
        results = pool.map(evaluate_model, model_args_list)
```

### 2. Serialization-Safe Operations

The project ensures all data passed between processes is properly serializable:

```python
# Safe approach - all parameters are basic types
model_config = {
    'sequence_length': 10,
    'lstm_units': 50,
    'dropout_rate': 0.2,
    'epochs': 100
}

# Pass configuration instead of model objects
task_args = (LSTMModel, model_config, series_data, steps)
```

## Running Parallel Experiments

### Basic Parallel Execution

```bash
# Run with synthetic data using all CPU cores
python -m forecasting.main --mode synthetic --parallel

# Run with specific number of processes
python -m forecasting.main --mode synthetic --parallel --num-processes 4

# Run with M4 data in parallel
python -m forecasting.main --mode m4 --parallel --max-series 20
```

### Advanced Parallel Usage

```python
# Example of custom parallel evaluation
from multiprocessing import Pool
import os

def run_model_evaluation(model_info):
    """Evaluate a single model configuration"""
    model_class, params, data = model_info
    
    model = model_class(**params)
    predictions = model.fit_predict(data['train'], len(data['test']))
    
    # Calculate metrics
    from forecasting.utils.metrics import calculate_metrics
    metrics = calculate_metrics(data['test'], predictions)
    
    return {
        'model_name': model_class.__name__,
        'metrics': metrics,
        'predictions': predictions
    }

if __name__ == "__main__":
    # Set spawn method for Mac compatibility
    if os.name == 'posix':
        import multiprocessing as mp
        mp.set_start_method('spawn', force=True)
    
    # Prepare tasks
    model_configs = [
        (LSTMModel, {'epochs': 50}, dataset),
        (ARIMAModel, {}, dataset),
        (ETSModel, {}, dataset),
        # ... more configurations
    ]
    
    # Run in parallel
    with Pool() as pool:
        results = pool.map(run_model_evaluation, model_configs)
```

## Performance Considerations on Mac

### 1. Process Overhead

Spawning processes has higher overhead than forking on Linux. Consider:

- **Small datasets**: Serial processing might be faster due to process creation overhead
- **Large datasets**: Parallel processing provides significant speedup

### 2. Memory Usage

Each process maintains its own memory space:

```python
# Efficient: Share data references when possible
shared_data = shared_memory.Array('d', large_dataset)

# Better: Process data in chunks
def process_chunk(chunk_info):
    chunk_data, model_params = chunk_info
    # Process only the needed portion
```

### 3. Optimal Process Count

For best performance on Mac:

```python
import os
import multiprocessing as mp

def get_optimal_processes():
    cpu_count = mp.cpu_count()
    
    # For CPU-intensive tasks like deep learning
    optimal = min(cpu_count, 8)  # Don't exceed 8 for intensive tasks
    
    # Adjust based on your specific workload
    if is_cpu_intensive_task():
        optimal = max(1, cpu_count // 2)
    
    return optimal
```

## Troubleshooting Common Issues

### 1. Pickle Errors

If you encounter pickle-related errors:

```python
# Problem: Trying to pass complex objects
problematic_arg = lambda x: x**2  # Lambdas can't be pickled

# Solution: Use named functions
def square(x):
    return x**2

# Or use functools.partial for parameterized functions
from functools import partial
param_func = partial(pow, exp=2)
```

### 2. Shared State Issues

Avoid shared state between processes:

```python
# Problem: Global variables aren't shared between processes
global_counter = 0

def worker():
    global global_counter
    global_counter += 1  # This won't affect parent process

# Solution: Return values instead
def worker_with_return(input_data):
    result = process(input_data)
    return result  # Return computed values
```

## Best Practices for Mac Parallel Processing

1. **Always use `if __name__ == "__main__":` guard**
2. **Set multiprocessing method explicitly on Mac**
3. **Keep data chunks reasonably sized**
4. **Handle exceptions within worker processes**
5. **Monitor memory usage during parallel execution**

```python
import multiprocessing as mp
import os
import sys

def safe_worker_function(args):
    """Worker function with proper error handling"""
    try:
        # Your processing logic here
        result = process_data(args)
        return {'status': 'success', 'result': result}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    # Essential for Mac compatibility
    if os.name == 'posix':
        mp.set_start_method('spawn', force=True)
    
    # Your parallel processing code here
    with mp.Pool() as pool:
        results = pool.map(safe_worker_function, task_list)
```

With these practices, you can efficiently utilize parallel processing on Mac systems while maintaining stability and performance.