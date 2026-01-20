# Configuration for reduced memory usage
import os
from pathlib import Path

class SmallConfig:
    # Basic settings
    TEST_SIZE = 6  # Reduced test size
    MAX_SERIES_COUNT = 2  # Only analyze first 2 series
    
    # LSTM settings - reduced to minimum
    LSTM_SEQUENCE_LENGTH = 5  # Very short sequences
    LSTM_EPOCHS = 1  # Minimal epochs
    LSTM_BATCH_SIZE = 1  # Smallest possible batch
    
    # Results directory
    RESULTS_DIR = Path(__file__).parent / "results_small"
    
    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)
    
    # Other settings
    SEASONALITY_THRESHOLD = 0.1  # Lower threshold
    CONFIDENCE_LEVEL = 0.95