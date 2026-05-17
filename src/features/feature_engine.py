"""
Feature Engineering — builds predictive features from OHLCV data
"""

import pandas as pd
import numpy as np
from typing import List, Optional

class FeatureEngine:
    """
    Multi-timeframe feature engineering for regime-aware forecasting.
    Features are calibrated for ensemble ML models (LightGBM/XGBoost/RF).
    """

    def __init__(self):
        self.windows = {
            "short": [5, 15, 30],
            "medium": [60, 240],
            "long": [1440]
        }

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build full feature set from raw OHLCV."""
        df = df.copy()

        # ─── Price returns ─────────────────────────────────────
        for window in [1, 5, 15, 60]:
            df[f"return_{window}m"] = df["close"].pct_change(window)

        # ─── Moving averages ───────────────────────────────────
        for window in [5, 20, 50, 200]:
            df[f"sma_{window}"] = df["close"].rolling(window).mean()
            df[f"ema_{window}"] = df["close"].ewm(span=window, adjust=False).mean()
            df[f"distance_to_sma_{window}"] = (df["close"] - df[f"sma_{window}"]) / df[f"sma_{window}"]

        # ─── Volatility ─────────────────────────────────────────
        for window in [15, 60, 1440]:
            df[f"volatility_{window}m"] = df["return_1m"].rolling(window).std()
            df[f"volatility_ratio_{window}m"] = df[f"volatility_{window}m"] / df[f"volatility_{window}m"].rolling(1440).mean().replace(0, 1e-8)

        # ─── Momentum indicators ────────────────────────────────
        df["rsi"] = self._compute_rsi(df["close"], window=14)
        df["macd"] = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # Stochastic
        low_min = df["low"].rolling(14).min()
        high_max = df["high"].rolling(14).max()
        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-8)
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # ─── Volume features ────────────────────────────────────
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, 1e-8)
        df["price_volume_correlation"] = df["close"].rolling(20).corr(df["volume"])

        # ─── Bollinger Bands ────────────────────────────────────
        df["bb_mid"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-8)

        # ─── Trend features ─────────────────────────────────────
        df["trend_angle"] = np.arctan(
            (df["close"].diff(20) / 20) / (df["close"] / 100 + 1e-8)
        ) * 180 / np.pi

        df["higher_highs"] = (df["high"] > df["high"].shift(1)).rolling(10).sum()
        df["lower_lows"] = (df["low"] < df["low"].shift(1)).rolling(10).sum()

        # ─── Support/Resistance ─────────────────────────────────
        df["rolling_high_100"] = df["high"].rolling(100).max()
        df["rolling_low_100"] = df["low"].rolling(100).min()
        df["distance_to_rolling_high"] = (df["close"] - df["rolling_high_100"]) / df["rolling_high_100"]
        df["distance_to_rolling_low"] = (df["close"] - df["rolling_low_100"]) / df["rolling_low_100"]

        # ─── Lag features ───────────────────────────────────────
        for lag in [1, 2, 3, 5, 10]:
            df[f"return_lag_{lag}"] = df["return_1m"].shift(lag)
            df[f"volume_ratio_lag_{lag}"] = df["volume_ratio"].shift(lag)

        # ─── Rolling statistics ─────────────────────────────────
        for window in [10, 30, 60]:
            df[f"skewness_{window}m"] = df["return_1m"].rolling(window).skew()
            df[f"kurtosis_{window}m"] = df["return_1m"].rolling(window).apply(
                lambda x: pd.Series(x).kurtosis() if len(x) > 3 else 0
            )
            df[f"max_drawdown_{window}m"] = (
                df["close"].rolling(window).apply(lambda x: (x / x.cummax() - 1).min(), raw=False)
            )

        # ─── Time features ──────────────────────────────────────
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["minute_of_hour"] = df["timestamp"].dt.minute

        # Drop NaN rows
        df = df.dropna()
        df = df.reset_index(drop=True)

        # Feature list for model use
        df["_feature_count"] = len([c for c in df.columns if c not in [
            "timestamp", "open", "high", "low", "close", "volume"
        ]])

        return df

    @staticmethod
    def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))