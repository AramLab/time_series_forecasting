"""
Модули для анализа данных и результатов прогнозирования.

Содержит:
- Анализ синтетических временных рядов
- Анализ данных M3 и M4
- Сравнение результатов моделей
"""

from .synthetic_data_analysis import SyntheticDataAnalysis
from .m3_m4_analysis import M3M4Analysis

__all__ = [
    'SyntheticDataAnalysis',
    'M3M4Analysis'
]