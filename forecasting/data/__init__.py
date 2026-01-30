"""
Data module initialization
"""

from .loader import DataLoader
from .synthetic_generator import SyntheticGenerator

__all__ = ['DataLoader', 'SyntheticGenerator']