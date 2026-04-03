"""
ML Data Pipeline - Historical Data Ingestion & Feature Engineering
═══════════════════════════════════════════════════════════════════════════════

This module handles:
1. Historical data ingestion (years of OHLCV data)
2. Feature engineering for ML models
3. Model training and saving
4. Live prediction interface

Architecture:
- OFFLINE: Train on historical data (run weekly/monthly)
- LIVE: Load trained model, predict on real-time features

Features extracted:
- Price-based: returns, volatility, momentum, mean reversion
- Volume-based: volume profile, relative volume, accumulation
- Technical: RSI, MACD, Bollinger, ATR, ADX
- Time-based: day of week, time of day, seasonality
- Market context: Nifty correlation, sector performance, VIX
"""

import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Directories
DATA_DIR = Path("data/historical")
MODEL_DIR = Path("models")
FEATURES_DIR = Path("data/features")

for d in [DATA_DIR, MODEL_DIR, FEATURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class FeatureEngineer:
    """
    Extracts features from OHLCV data for ML models.
    
    Feature Categories:
    1. PRICE FEATURES (what price is doing)
    2. VOLUME FEATURES (where smart money is)
    3. TECHNICAL INDICATORS (classic signals)
    4. PATTERN FEATURES (candle patterns, S/R)
    5. CONTEXT FEATURES (market, sector, time)
    """
    
    def __init__(self):
        self.feature_names = []
    
    def compute_all_features(self, df: pd.DataFrame, 
                             market_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Compute all features for a stock DataFrame.
        
        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
            market_df: Optional Nifty/market index data for correlation
        
        Returns:
            DataFrame with all features added
        """
        df = df.copy()
        
        # Ensure datetime index
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        
        # 1. PRICE FEATURES
        df = self._add_price_features(df)
        
        # 2. VOLUME FEATURES
        df = self._add_volume_features(df)
        
        # 3. TECHNICAL INDICATORS
        df = self._add_technical_indicators(df)
        
        # 4. PATTERN FEATURES
        df = self._add_pattern_features(df)
        
        # 5. CONTEXT FEATURES
        df = self._add_context_features(df, market_df)
        
        # Store feature names
        self.feature_names = [c for c in df.columns if c.startswith('f_')]
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-based features."""
        # Returns at various horizons
        for period in [1, 3, 5, 10, 20]:
            df[f'f_return_{period}'] = df['close'].pct_change(period)
        
        # Volatility (rolling std of returns)
        df['f_volatility_5'] = df['close'].pct_change().rolling(5).std()
        df['f_volatility_20'] = df['close'].pct_change().rolling(20).std()
        
        # Intraday range
        df['f_range_pct'] = (df['high'] - df['low']) / df['close']
        df['f_range_avg_5'] = df['f_range_pct'].rolling(5).mean()
        
        # Position in range (where close is within H-L)
        df['f_close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Gap from previous close
        df['f_gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        
        # Distance from highs/lows
        df['f_dist_from_high_5'] = (df['close'] - df['high'].rolling(5).max()) / df['close']
        df['f_dist_from_low_5'] = (df['close'] - df['low'].rolling(5).min()) / df['close']
        df['f_dist_from_high_20'] = (df['close'] - df['high'].rolling(20).max()) / df['close']
        df['f_dist_from_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
        
        # Momentum
        df['f_momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['f_momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['f_momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Mean reversion signal (deviation from moving average)
        df['f_ma_dev_5'] = (df['close'] - df['close'].rolling(5).mean()) / df['close'].rolling(5).mean()
        df['f_ma_dev_20'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).mean()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features."""
        # Relative volume
        df['f_rel_volume_5'] = df['volume'] / df['volume'].rolling(5).mean()
        df['f_rel_volume_20'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Volume trend
        df['f_volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        # Price-volume divergence (price up, volume down = weak)
        price_change = df['close'].pct_change(5)
        volume_change = df['volume'].pct_change(5)
        df['f_pv_divergence'] = price_change - volume_change
        
        # Accumulation/Distribution proxy
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        df['f_acc_dist'] = (clv * df['volume']).rolling(10).sum()
        df['f_acc_dist_norm'] = df['f_acc_dist'] / df['f_acc_dist'].rolling(20).std()
        
        # Volume at price levels (simplified VWAP deviation)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
        df['f_vwap_dev'] = (df['close'] - vwap) / df['close']
        
        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classic technical indicators."""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['f_rsi_14'] = 100 - (100 / (1 + rs))
        df['f_rsi_norm'] = (df['f_rsi_14'] - 50) / 50  # Normalized -1 to 1
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        df['f_macd'] = macd / df['close']  # Normalized
        df['f_macd_signal'] = signal / df['close']
        df['f_macd_hist'] = (macd - signal) / df['close']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['f_bb_upper'] = (sma_20 + 2 * std_20 - df['close']) / df['close']
        df['f_bb_lower'] = (df['close'] - (sma_20 - 2 * std_20)) / df['close']
        df['f_bb_width'] = (4 * std_20) / sma_20  # Volatility measure
        df['f_bb_position'] = (df['close'] - (sma_20 - 2 * std_20)) / (4 * std_20)
        
        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['f_atr_14'] = tr.rolling(14).mean() / df['close']  # Normalized
        
        # ADX (simplified)
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff().abs()
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
        
        tr_14 = tr.rolling(14).sum()
        plus_di = 100 * (plus_dm.rolling(14).sum() / tr_14)
        minus_di = 100 * (minus_dm.rolling(14).sum() / tr_14)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df['f_adx'] = dx.rolling(14).mean() / 100  # Normalized 0-1
        df['f_di_diff'] = (plus_di - minus_di) / 100  # Direction
        
        # EMA crossover signals
        ema_9 = df['close'].ewm(span=9).mean()
        ema_21 = df['close'].ewm(span=21).mean()
        df['f_ema_cross'] = (ema_9 - ema_21) / df['close']
        df['f_ema_cross_signal'] = np.sign(df['f_ema_cross']) - np.sign(df['f_ema_cross'].shift(1))
        
        return df
    
    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick and price pattern features."""
        # Candle body size
        df['f_body_size'] = (df['close'] - df['open']).abs() / df['close']
        df['f_body_direction'] = np.sign(df['close'] - df['open'])
        
        # Upper/lower shadows
        df['f_upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['f_lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # Doji detection (small body, big shadows)
        df['f_is_doji'] = (df['f_body_size'] < df['f_range_pct'] * 0.2).astype(int)
        
        # Engulfing pattern (simplified)
        prev_body = (df['close'].shift(1) - df['open'].shift(1))
        curr_body = (df['close'] - df['open'])
        df['f_engulfing'] = ((curr_body * prev_body < 0) & 
                            (curr_body.abs() > prev_body.abs())).astype(int) * np.sign(curr_body)
        
        # Consecutive candles in same direction
        direction = np.sign(df['close'] - df['open'])
        df['f_consecutive_dir'] = direction.groupby((direction != direction.shift()).cumsum()).cumcount() + 1
        df['f_consecutive_dir'] *= direction
        
        # Support/Resistance proximity (simplified)
        recent_highs = df['high'].rolling(20).max()
        recent_lows = df['low'].rolling(20).min()
        df['f_near_resistance'] = (recent_highs - df['close']) / df['close']
        df['f_near_support'] = (df['close'] - recent_lows) / df['close']
        
        return df
    
    def _add_context_features(self, df: pd.DataFrame, 
                              market_df: pd.DataFrame = None) -> pd.DataFrame:
        """Market context and time features."""
        # Time features (if datetime index)
        if isinstance(df.index, pd.DatetimeIndex):
            df['f_day_of_week'] = df.index.dayofweek / 4  # Normalized 0-1
            df['f_month'] = df.index.month / 12
            
            # For intraday data
            if len(df) > 100:  # Likely intraday
                df['f_hour'] = df.index.hour / 24
                df['f_minute_norm'] = df.index.minute / 60
                
                # Time of day buckets
                hour = df.index.hour
                df['f_is_open'] = ((hour == 9) | ((hour == 10) & (df.index.minute < 30))).astype(int)
                df['f_is_close'] = ((hour == 15) | ((hour == 14) & (df.index.minute > 30))).astype(int)
                df['f_is_lunch'] = ((hour >= 12) & (hour < 14)).astype(int)
        
        # Market correlation (if market data provided)
        if market_df is not None and len(market_df) == len(df):
            market_ret = market_df['close'].pct_change()
            stock_ret = df['close'].pct_change()
            df['f_market_corr_20'] = stock_ret.rolling(20).corr(market_ret)
            df['f_market_beta'] = stock_ret.rolling(20).cov(market_ret) / market_ret.rolling(20).var()
            df['f_alpha'] = stock_ret - df['f_market_beta'] * market_ret
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature column names."""
        return self.feature_names


class MLDataPipeline:
    """
    Complete ML data pipeline for trading.
    
    OFFLINE PHASE:
    1. Load historical data
    2. Engineer features
    3. Create target labels
    4. Train models
    5. Save models
    
    LIVE PHASE:
    1. Load trained model
    2. Compute features on live data
    3. Predict
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.model = None
        self.scaler = None
        self.feature_names = []
    
    # ════════════════════════════════════════════════════════════
    # OFFLINE: Training Pipeline
    # ════════════════════════════════════════════════════════════
    
    def load_historical_data(self, symbol: str, 
                             start_date: str = "2020-01-01",
                             end_date: str = None) -> pd.DataFrame:
        """
        Load historical OHLCV data for training.
        
        In production, this would fetch from:
        - Local database (TimescaleDB)
        - Data provider API (Angel One historical)
        - CSV files
        """
        end_date = end_date or str(date.today())
        
        # Check for cached data
        cache_file = DATA_DIR / f"{symbol}_{start_date}_{end_date}.parquet"
        if cache_file.exists():
            logger.info(f"Loading cached data for {symbol}")
            return pd.read_parquet(cache_file)
        
        # Try to fetch from yfinance as fallback
        try:
            import yfinance as yf
            ticker = f"{symbol}.NS"
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={'adj close': 'adj_close'})
            
            # Cache for future use
            df.to_parquet(cache_file)
            logger.info(f"Loaded {len(df)} days of data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data for {symbol}: {e}")
            return pd.DataFrame()
    
    def create_training_dataset(self, symbols: List[str],
                                 start_date: str = "2020-01-01",
                                 end_date: str = None,
                                 forward_horizon: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create training dataset with features and labels.
        
        Args:
            symbols: List of stock symbols
            start_date: Training start date
            end_date: Training end date
            forward_horizon: Days ahead to predict (for label)
        
        Returns:
            X: Feature DataFrame
            y: Target labels (1 = up, 0 = down)
        """
        all_features = []
        all_labels = []
        
        # Load Nifty for market context
        nifty_df = self.load_historical_data("NIFTY", start_date, end_date)
        
        for symbol in symbols:
            df = self.load_historical_data(symbol, start_date, end_date)
            if df.empty:
                continue
            
            # Compute features
            df = self.feature_engineer.compute_all_features(df, nifty_df)
            
            # Create label: forward return classification
            forward_return = df['close'].shift(-forward_horizon) / df['close'] - 1
            
            # Binary classification: 1 if return > 0.5%, else 0
            # Can adjust threshold based on transaction costs
            label = (forward_return > 0.005).astype(int)
            
            # Drop rows with NaN features or labels
            valid_idx = df[self.feature_engineer.feature_names].dropna().index
            valid_idx = valid_idx.intersection(label.dropna().index)
            
            features = df.loc[valid_idx, self.feature_engineer.feature_names]
            labels = label.loc[valid_idx]
            
            # Add symbol column for reference
            features['symbol'] = symbol
            
            all_features.append(features)
            all_labels.append(labels)
        
        if not all_features:
            raise ValueError("No valid training data created")
        
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)
        
        # Store feature names (excluding symbol)
        self.feature_names = [c for c in X.columns if c != 'symbol']
        
        logger.info(f"Created training dataset: {len(X)} samples, {len(self.feature_names)} features")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series,
                    model_type: str = "lightgbm",
                    use_early_stopping: bool = True) -> dict:
        """
        Train ML model on features with early stopping and learning rate scheduling.
        
        Args:
            X: Feature DataFrame
            y: Target labels
            model_type: "lightgbm", "xgboost", "random_forest", or "neural_net"
            use_early_stopping: Enable early stopping for boosting models
        
        Returns:
            Dict with model metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Remove symbol column for training
        feature_cols = [c for c in X.columns if c != 'symbol']
        X_train_full = X[feature_cols]
        
        # Train-validation-test split (for early stopping)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_train_full, y, test_size=0.2, shuffle=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, shuffle=False
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        early_stopping_rounds = 50 if use_early_stopping else None
        best_iteration = None
        
        # Train model
        if model_type == "lightgbm":
            try:
                import lightgbm as lgb
                
                callbacks = []
                if use_early_stopping:
                    callbacks = [
                        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
                        lgb.log_evaluation(period=50)
                    ]
                
                self.model = lgb.LGBMClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=31,
                    min_child_samples=20,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    verbose=-1
                )
                
                self.model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    callbacks=callbacks if callbacks else None
                )
                best_iteration = self.model.best_iteration_ if use_early_stopping else None
                
            except ImportError:
                logger.warning("LightGBM not available, falling back to RandomForest")
                model_type = "random_forest"
        
        if model_type == "xgboost":
            try:
                import xgboost as xgb
                
                self.model = xgb.XGBClassifier(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    early_stopping_rounds=early_stopping_rounds if use_early_stopping else None,
                    random_state=42,
                    eval_metric='logloss'
                )
                
                self.model.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    verbose=50
                )
                best_iteration = self.model.best_iteration if use_early_stopping else None
                
            except ImportError:
                logger.warning("XGBoost not available, falling back to RandomForest")
                model_type = "random_forest"
        
        if model_type == "neural_net":
            self.model, best_iteration = self._train_neural_net(
                X_train_scaled, y_train,
                X_val_scaled, y_val,
                use_early_stopping
            )
        
        if model_type == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        metrics = {
            "model_type": model_type,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y_test, y_proba),
            "best_iteration": best_iteration,
            "early_stopping_used": use_early_stopping
        }
        
        # Feature importance (not available for neural_net)
        if hasattr(self.model, 'feature_importances_'):
            metrics["feature_importance"] = dict(zip(
                feature_cols,
                self.model.feature_importances_.tolist()
            ))
        
        logger.info(f"Model trained: Accuracy={metrics['accuracy']:.3f}, AUC={metrics['auc_roc']:.3f}")
        if best_iteration:
            logger.info(f"Best iteration: {best_iteration}")
        
        return metrics
    
    def _train_neural_net(self, X_train, y_train, X_val, y_val, 
                          use_early_stopping: bool = True):
        """
        Train neural network with learning rate scheduler.
        
        Uses ReduceLROnPlateau scheduler that reduces LR when validation
        loss plateaus, plus optional early stopping.
        """
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
            from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
        except ImportError:
            logger.warning("PyTorch not available, falling back to RandomForest")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            return model, None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training neural network on {device}")
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train.values).to(device)
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val.values).to(device)
        
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        
        # Define network
        input_dim = X_train.shape[1]
        
        class TradingNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.net(x).squeeze()
            
            def predict(self, X):
                self.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X).to(next(self.parameters()).device)
                    return (self.forward(X_t).cpu().numpy() > 0.5).astype(int)
            
            def predict_proba(self, X):
                self.eval()
                with torch.no_grad():
                    X_t = torch.FloatTensor(X).to(next(self.parameters()).device)
                    prob_1 = self.forward(X_t).cpu().numpy()
                    return np.column_stack([1 - prob_1, prob_1])
        
        model = TradingNet(input_dim).to(device)
        
        # Optimizer with initial learning rate
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # Learning rate schedulers
        # 1. ReduceLROnPlateau: reduce LR when val loss plateaus
        scheduler_plateau = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, 
            verbose=True, min_lr=1e-6
        )
        # 2. Cosine annealing with warm restarts (optional alternative)
        # scheduler_cosine = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        criterion = nn.BCELoss()
        
        # Training loop
        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        early_stop_patience = 30
        
        epochs = 200
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_t)
                val_loss = criterion(val_output, y_val_t).item()
            
            # Update scheduler based on validation loss
            scheduler_plateau.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Logging
            if epoch % 20 == 0:
                train_loss /= len(train_loader)
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                           f"val_loss={val_loss:.4f}, lr={current_lr:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if use_early_stopping and patience_counter >= early_stop_patience:
                logger.info(f"Early stopping at epoch {epoch}, best was {best_epoch}")
                break
        
        # Load best model
        model.load_state_dict(best_state)
        model.eval()
        
        return model, best_epoch
    
    def save_model(self, model_name: str = "trading_model"):
        """Save trained model and scaler."""
        if self.model is None:
            raise ValueError("No model to save")
        
        model_path = MODEL_DIR / f"{model_name}.pkl"
        scaler_path = MODEL_DIR / f"{model_name}_scaler.pkl"
        meta_path = MODEL_DIR / f"{model_name}_meta.json"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        meta = {
            "feature_names": self.feature_names,
            "created_at": datetime.now().isoformat(),
            "model_type": type(self.model).__name__
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
    
    # ════════════════════════════════════════════════════════════
    # LIVE: Prediction Pipeline
    # ════════════════════════════════════════════════════════════
    
    def load_model(self, model_name: str = "trading_model") -> bool:
        """Load trained model for live prediction."""
        model_path = MODEL_DIR / f"{model_name}.pkl"
        scaler_path = MODEL_DIR / f"{model_name}_scaler.pkl"
        meta_path = MODEL_DIR / f"{model_name}_meta.json"
        
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return False
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            self.feature_names = meta["feature_names"]
        
        logger.info(f"Model loaded: {meta['model_type']} with {len(self.feature_names)} features")
        return True
    
    def predict_live(self, live_df: pd.DataFrame) -> Dict:
        """
        Make prediction on live data.
        
        Args:
            live_df: Recent OHLCV data (at least 30 candles for features)
        
        Returns:
            Dict with prediction, probability, and confidence
        """
        if self.model is None:
            if not self.load_model():
                return {"error": "No model loaded"}
        
        # Compute features on live data
        df = self.feature_engineer.compute_all_features(live_df)
        
        # Get latest row features
        latest = df[self.feature_names].iloc[-1:].values
        
        # Handle missing features
        if np.any(np.isnan(latest)):
            # Fill NaN with 0 (neutral value for normalized features)
            latest = np.nan_to_num(latest, nan=0.0)
        
        # Scale
        latest_scaled = self.scaler.transform(latest)
        
        # Predict
        prob = self.model.predict_proba(latest_scaled)[0]
        pred = self.model.predict(latest_scaled)[0]
        
        # Confidence = distance from 0.5
        confidence = abs(prob[1] - 0.5) * 2  # 0 to 1
        
        return {
            "prediction": int(pred),  # 1 = bullish, 0 = bearish
            "prob_up": float(prob[1]),
            "prob_down": float(prob[0]),
            "confidence": float(confidence),
            "signal": "LONG" if prob[1] > 0.55 else ("SHORT" if prob[1] < 0.45 else "NEUTRAL"),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_feature_importance(self, top_n: int = 10) -> Dict:
        """Get top important features."""
        if self.model is None:
            return {}
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_imp[:top_n])


# ════════════════════════════════════════════════════════════
# UTILITIES for live_paper_v3.py integration
# ════════════════════════════════════════════════════════════

def train_trading_model(symbols: List[str] = None,
                        start_date: str = "2022-01-01",
                        model_name: str = "trading_model",
                        model_type: str = "lightgbm",
                        use_early_stopping: bool = True):
    """
    Train ML model on historical data.
    Run this OFFLINE (weekly/monthly).
    
    Args:
        symbols: List of stock symbols to train on
        start_date: Training start date
        model_name: Name for saved model
        model_type: "lightgbm", "xgboost", "random_forest", or "neural_net"
        use_early_stopping: Enable early stopping (recommended)
    
    Example:
        python -m ml.train_model --model-type neural_net
    """
    if symbols is None:
        # Default: Nifty 50 stocks
        symbols = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
            "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
            "BAJFINANCE", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
            "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO", "HCLTECH"
        ]
    
    pipeline = MLDataPipeline()
    
    print(f"Creating training dataset from {start_date}...")
    X, y = pipeline.create_training_dataset(symbols, start_date)
    
    print(f"Training {model_type} model...")
    metrics = pipeline.train_model(X, y, model_type=model_type, 
                                   use_early_stopping=use_early_stopping)
    
    print(f"\n=== MODEL METRICS ===")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.3f}")
    if metrics.get('best_iteration'):
        print(f"Best Iteration: {metrics['best_iteration']}")
    
    if 'feature_importance' in metrics:
        print(f"\n=== TOP 10 FEATURES ===")
        top_features = sorted(metrics['feature_importance'].items(), 
                             key=lambda x: x[1], reverse=True)[:10]
        for feat, imp in top_features:
            print(f"  {feat}: {imp:.4f}")
    
    print(f"\nSaving model as '{model_name}'...")
    pipeline.save_model(model_name)
    
    return metrics


def get_ml_prediction(symbol: str, live_df: pd.DataFrame) -> Dict:
    """
    Get ML prediction for live trading.
    Call this from live_paper_v3.py.
    
    Args:
        symbol: Stock symbol
        live_df: Recent OHLCV data (last 50+ candles)
    
    Returns:
        Dict with signal, probability, confidence
    """
    pipeline = MLDataPipeline()
    return pipeline.predict_live(live_df)


if __name__ == "__main__":
    # Quick test
    print("Testing ML Data Pipeline...")
    
    # Test feature engineering
    fe = FeatureEngineer()
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    sample_df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(100000, 1000000, 100),
    }, index=dates)
    sample_df['high'] = sample_df[['open', 'close', 'high']].max(axis=1)
    sample_df['low'] = sample_df[['open', 'close', 'low']].min(axis=1)
    
    df_features = fe.compute_all_features(sample_df)
    print(f"\nFeatures computed: {len(fe.feature_names)}")
    print(f"Sample features: {fe.feature_names[:10]}")
    
    print("\n✅ ML Data Pipeline ready!")
    print("\nTo train model on historical data:")
    print("  python -m ml.data_pipeline")
