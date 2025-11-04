"""
Machine learning models for AI Trading Bot.

Provides LSTM, GRU and optional GPT integration for price prediction and decision making.
"""

from .lstm import LSTMPredictor

# Optional GPT integration
try:
    from .gpt import GPTIntegration
except ImportError:
    GPTIntegration = None

# Optional GRU predictor
try:
    from .gru_predictor import GRUPricePredictor
except ImportError:
    GRUPricePredictor = None

__all__ = [
    "LSTMPredictor",
    "GPTIntegration",
    "GRUPricePredictor"
]