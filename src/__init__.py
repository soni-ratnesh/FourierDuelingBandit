"""
A Fourier Approach to Sample-Efficient Copeland Bandit Identification

This library implements Fourier-sparse methods for efficient Copeland Winner
identification, comparing standard O(N²) approaches with O(k log N) methods.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_generator import generate_preference_matrix
from .naive_ccb import NaiveCCB
from .fourier_bandit import FourierDuelingBandit
from .sushi_loader import (
    load_sushi_dataset,
    generate_synthetic_sushi_preferences,
    get_sushi_features
)

__all__ = [
    "generate_preference_matrix",
    "NaiveCCB", 
    "FourierDuelingBandit",
    "load_sushi_dataset",
    "generate_synthetic_sushi_preferences",
    "get_sushi_features"
]
