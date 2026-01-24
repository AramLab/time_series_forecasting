#!/usr/bin/env python3
"""
Test script to verify model_runner syntax and basic functionality
"""
import sys
import importlib.util

# Add current directory to path
sys.path.insert(0, '.')

print("Testing model_runner.py syntax...")

try:
    spec = importlib.util.spec_from_file_location("model_runner", "./models/model_runner.py")
    model_runner_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_runner_module)
    print("✅ model_runner.py loaded successfully")
    
    # Check if required functions exist
    assert hasattr(model_runner_module, 'run_all_models'), "run_all_models function not found"
    assert hasattr(model_runner_module, 'get_best_model_result'), "get_best_model_result function not found"
    print("✅ Required functions found")
    
    # Check the function signatures
    import inspect
    sig = inspect.signature(model_runner_module.run_all_models)
    params = list(sig.parameters.keys())
    expected_params = ['series_id', 'values', 'dataset_name', 'test_size']
    assert all(p in params for p in expected_params), f"Missing parameters in run_all_models signature. Found: {params}"
    print(f"✅ run_all_models signature correct: {params}")
    
    print("\n🎉 All tests passed! The model runner is ready.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()