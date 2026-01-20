import os
from pathlib import Path


class Config:
    # Пути к директориям
    BASE_DIR = Path(__file__).parent.parent
    RESULTS_DIR = BASE_DIR / "results"
    DATA_DIR = BASE_DIR / "data"

    # Параметры по умолчанию
    TEST_SIZE = 12
    RANDOM_SEED = 42

    # Параметры для моделей
    LSTM_SEQUENCE_LENGTH = 12
    LSTM_EPOCHS = 50
    LSTM_BATCH_SIZE = 16

    # Параметры для CEEMDAN
    CEEMDAN_TRIALS = 20
    CEEMDAN_NOISE_WIDTH = 0.05

    @classmethod
    def setup_directories(cls):
        """Создание необходимых директорий"""
        os.makedirs(cls.RESULTS_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)