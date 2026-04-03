#!/usr/bin/env python3
"""
ML Model Training Script
═══════════════════════════════════════════════════════════════════════════════

Run this script OFFLINE to train/retrain ML models on historical data.

Schedule: Run weekly (weekends) or monthly

Usage:
    python -m ml.train_model                     # Default: last 2 years
    python -m ml.train_model --start 2023-01-01  # Custom start date
    python -m ml.train_model --symbols RELIANCE TCS HDFCBANK
"""

import argparse
from datetime import date
from data_pipeline import train_trading_model, MLDataPipeline

# Default stocks (Nifty 50 + key FNO stocks)
DEFAULT_SYMBOLS = [
    # IT
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTI",
    # Banks
    "HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK",
    # Finance
    "BAJFINANCE", "BAJAJFINSV", "HDFC", "SBILIFE", "HDFCLIFE",
    # FMCG
    "HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO",
    # Auto
    "MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO",
    # Pharma
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
    # Industrial
    "RELIANCE", "LT", "ONGC", "NTPC", "POWERGRID", "COALINDIA",
    # Others
    "ASIANPAINT", "TITAN", "ULTRACEMCO", "GRASIM", "ADANIENT",
    "BHARTIARTL", "JSWSTEEL", "TATASTEEL", "HINDALCO",
]


def main():
    parser = argparse.ArgumentParser(description="Train ML model on historical data")
    parser.add_argument("--start", type=str, default="2022-01-01",
                       help="Training start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None,
                       help="Training end date (default: today)")
    parser.add_argument("--symbols", nargs="+", default=None,
                       help="List of stock symbols")
    parser.add_argument("--model-name", type=str, default="trading_model",
                       help="Name for saved model")
    parser.add_argument("--model-type", type=str, default="lightgbm",
                       choices=["random_forest", "lightgbm", "xgboost", "neural_net"],
                       help="ML model type")
    parser.add_argument("--no-early-stopping", action="store_true",
                       help="Disable early stopping")
    
    args = parser.parse_args()
    
    symbols = args.symbols or DEFAULT_SYMBOLS
    use_early_stopping = not args.no_early_stopping
    
    print("=" * 70)
    print("  ML MODEL TRAINING")
    print("=" * 70)
    print(f"  Start Date: {args.start}")
    print(f"  End Date: {args.end or 'today'}")
    print(f"  Symbols: {len(symbols)} stocks")
    print(f"  Model Type: {args.model_type}")
    print(f"  Early Stopping: {'Enabled' if use_early_stopping else 'Disabled'}")
    if args.model_type == "neural_net":
        print(f"  LR Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)")
    print(f"  Output: models/{args.model_name}.pkl")
    print("=" * 70)
    print()
    
    # Train
    metrics = train_trading_model(
        symbols=symbols,
        start_date=args.start,
        model_name=args.model_name,
        model_type=args.model_type,
        use_early_stopping=use_early_stopping
    )
    
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"""
  Model saved to: models/{args.model_name}.pkl
  
  To use in live trading:
    from ml import get_ml_prediction
    prediction = get_ml_prediction(symbol, live_df)
    
  Next steps:
    1. Run paper trading to validate
    2. Schedule weekly retraining (cron/Task Scheduler)
    3. Monitor model degradation over time
""")


if __name__ == "__main__":
    main()
