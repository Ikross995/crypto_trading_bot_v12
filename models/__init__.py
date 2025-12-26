"""
Machine learning models for AI Trading Bot.

Provides PyTorch GRU and optional GPT integration for price prediction and decision making.
"""

# LSTM disabled - use PyTorch GRU instead
# from .lstm import LSTMPredictor
LSTMPredictor = None

# Optional GPT integration
try:
    from .gpt import GPTIntegration
except ImportError:
    GPTIntegration = None

# PyTorch GRU predictor (NEW!)
try:
    from .gru_predictor_pytorch import GRUPredictorPyTorch
except ImportError:
    GRUPredictorPyTorch = None

# Old TensorFlow GRU (deprecated)
# try:
#     from .gru_predictor import GRUPricePredictor
# except ImportError:
GRUPricePredictor = None

__all__ = [
    "LSTMPredictor",
    "GPTIntegration",
    "GRUPredictorPyTorch",
    "GRUPricePredictor"
]