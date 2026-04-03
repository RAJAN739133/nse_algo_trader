"""
ML Module for NSE Algo Trader
═════════════════════════════

OFFLINE (Historical Data):
- data_pipeline.py: Feature engineering, model training

LIVE (Real-Time):
- Load trained models
- Predict on live features

Usage:
    # Train model (run weekly)
    python -m ml.train_model
    
    # In live trading
    from ml.data_pipeline import get_ml_prediction
    prediction = get_ml_prediction(symbol, live_df)
"""

from .data_pipeline import (
    FeatureEngineer,
    MLDataPipeline,
    train_trading_model,
    get_ml_prediction,
)

__all__ = [
    'FeatureEngineer',
    'MLDataPipeline', 
    'train_trading_model',
    'get_ml_prediction',
]
