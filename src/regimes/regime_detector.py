"""
Regime Detection — online market regime classifier
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field as dc_field
from typing import Dict, List
from datetime import datetime

@dataclass
class MarketContext:
    symbol: str = "BTC/USDT"
    current_price: float = 0.0
    timestamp: datetime = None
    regime: str = "normal"
    regime_confidence: float = 0.5
    volatility_zscore: float = 0.0
    trend_strength: float = 0.0
    volume_ratio: float = 1.0
    price_momentum: float = 0.0
    regime_history: List[Dict] = dc_field(default_factory=list)
    price_level: float = 0.0

class RegimeDetector:
    """
    Classifies market into: normal | volatile | trending | anomalous
    Uses composite z-score of volatility, trend, volume, and momentum indicators.
    """

    def __init__(self, lookback: int = 100):
        self.lookback = lookback

    def classify(self, df: pd.DataFrame, features: pd.DataFrame = None) -> MarketContext:
        """Classify the current market regime."""
        if len(df) < 30:
            return self._default_context(df)

        returns = df["close"].pct_change().dropna().values
        volumes = df["volume"].values

        # Volatility score
        vol_now = np.std(returns[-15:]) if len(returns) >= 15 else np.std(returns)
        vol_hist = np.std(returns[-self.lookback:]) if len(returns) >= self.lookback else np.std(returns)
        vol_score = (vol_now / (vol_hist + 1e-8)) - 1.0

        # Trend score
        recent = returns[-20:] if len(returns) >= 20 else returns
        trend_score = np.mean(recent) / (np.std(recent) + 1e-8)

        # Price level z-score
        lookback_data = df["close"].values[-self.lookback:]
        rolling_mean = np.mean(lookback_data)
        rolling_std = np.std(lookback_data) + 1e-8
        price_level = (df["close"].iloc[-1] - rolling_mean) / rolling_std

        # Volume ratio
        vol_ma = np.mean(volumes[-self.lookback:]) if len(volumes) >= self.lookback else np.mean(volumes)
        vol_ratio = volumes[-1] / (vol_ma + 1e-8)

        # Momentum divergence
        if len(returns) >= 30:
            momentum_div = np.mean(returns[-10:]) - np.mean(returns[-30:-10])
        else:
            momentum_div = 0.0

        # Classification
        if abs(vol_score) > 2.5 or vol_ratio > 4.0:
            regime_name = "anomalous"
            confidence = min(1.0, abs(vol_score) / 3.5)
        elif abs(price_level) > 3.0 or abs(vol_score) > 2.0:
            regime_name = "volatile"
            confidence = min(1.0, 0.5 + abs(vol_score) / 4.0)
        elif abs(trend_score) > 1.5 and abs(momentum_div) > 0.5:
            regime_name = "trending"
            confidence = min(1.0, abs(trend_score) / 2.5)
        else:
            regime_name = "normal"
            confidence = min(1.0, 0.6 + (1.0 - abs(vol_score)) * 0.1)

        ts = df["timestamp"].iloc[-1] if "timestamp" in df.columns and len(df) > 0 else datetime.utcnow()

        return MarketContext(
            symbol="BTC/USDT",
            current_price=float(df["close"].iloc[-1]),
            timestamp=ts,
            regime=regime_name,
            regime_confidence=float(confidence),
            volatility_zscore=float(vol_score),
            trend_strength=float(trend_score),
            volume_ratio=float(vol_ratio),
            price_momentum=float(momentum_div),
            price_level=float(price_level)
        )

    def _default_context(self, df: pd.DataFrame) -> MarketContext:
        """Return default context when insufficient data."""
        price = float(df["close"].iloc[-1]) if len(df) > 0 else 50000.0
        ts = df["timestamp"].iloc[-1] if len(df) > 0 and "timestamp" in df.columns else datetime.utcnow()
        return MarketContext(symbol="BTC/USDT", current_price=price, timestamp=ts)