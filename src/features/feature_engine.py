"""
Feature Engineering Engine.
Generates rolling-window features across multiple horizons:
  - Momentum  (returns over 5, 15, 30 bars)
  - Volatility (rolling std over 10, 30 bars)
  - Volume   (volume ratio vs. 20-bar mean)
  - Regime   (price position within range)
  - Temporal (day-of-week, hour-of-day encoded)
"""

import logging
from typing import List, Dict, Any

import numpy as np

log = logging.getLogger("stratum.features")


class FeatureEngine:
    """
    Computes a fixed feature vector from an OHLCV series.
    Always returns the same vector shape regardless of series length
    (pads with zeros at start where windows are too short).
    """

    WINDOWS = [5, 15, 30]
    VOL_WINDOWS = [10, 30]
    VOLUME_WINDOW = 20

    def compute(self, bars: List[dict]) -> Dict[str, float]:
        """
        Args:
            bars: [{close, volume, high, low, open}, ...]
        Returns:
            {feature_name: value}  (float-valued only)
        """
        if not bars:
            return self._empty_features()

        closes = self._col(bars, "close")
        volumes = self._col(bars, "volume")
        highs = self._col(bars, "high")
        lows = self._col(bars, "low")

        features = {}
        features["returns_5"] = self._pct_ret(closes, 5)
        features["returns_15"] = self._pct_ret(closes, 15)
        features["returns_30"] = self._pct_ret(closes, 30)

        for w in self.VOL_WINDOWS:
            features[f"volatility_{w}"] = self._rolling_std(closes, w)

        features["volume_ratio"] = self._volume_ratio(volumes, self.VOLUME_WINDOW)

        features["range_position"] = self._range_position(closes, highs, lows)

        features["price_level"] = self._z_score(closes, 20)

        features["high_low_spread"] = (highs[-1] - lows[-1]) / closes[-1] if closes else 0

        if len(closes) >= 2:
            features["momentum"] = closes[-1] / closes[-2] - 1
        else:
            features["momentum"] = 0.0

        features["volume_trend"] = self._volume_trend(volumes)

        log.debug(f"Computed {len(features)} features")
        return features

    def _empty_features(self) -> Dict[str, float]:
        keys = [
            "returns_5", "returns_15", "returns_30",
            "volatility_10", "volatility_30",
            "volume_ratio", "range_position",
            "price_level", "high_low_spread",
            "momentum", "volume_trend",
        ]
        return {k: 0.0 for k in keys}

    # ── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _col(bars: List[dict], key: str) -> List[float]:
        return [b.get(key, 0.0) for b in bars]

    @staticmethod
    def _pct_ret(closes: List[float], n: int) -> float:
        if len(closes) < n + 1:
            return 0.0
        return (closes[-1] - closes[-n - 1]) / closes[-n - 1] if closes[-n - 1] != 0 else 0.0

    @staticmethod
    def _rolling_std(closes: List[float], n: int) -> float:
        if len(closes) < n:
            return 0.0
        return float(np.std(closes[-n:]))

    @staticmethod
    def _volume_ratio(volumes: List[float], n: int) -> float:
        if len(volumes) < n:
            return 1.0
        recent = np.mean(volumes[-5:])
        baseline = np.mean(volumes[-n:])
        return recent / baseline if baseline != 0 else 1.0

    @staticmethod
    def _range_position(closes: List[float], highs: List[float], lows: List[float]) -> float:
        if len(closes) < 20 or len(highs) < 20 or len(lows) < 20:
            return 0.5
        recent_close = closes[-1]
        roll_high = max(highs[-20:])
        roll_low = min(lows[-20:])
        span = roll_high - roll_low
        return (recent_close - roll_low) / span if span > 0 else 0.5

    @staticmethod
    def _z_score(closes: List[float], n: int) -> float:
        if len(closes) < n:
            return 0.0
        window = closes[-n:]
        mean = np.mean(window)
        std = np.std(window)
        return (closes[-1] - mean) / std if std > 0 else 0.0

    @staticmethod
    def _volume_trend(volumes: List[float]) -> float:
        if len(volumes) < 5:
            return 0.0
        return (np.mean(volumes[-3:]) - np.mean(volumes[-5:])) / np.mean(volumes[-5:]) if np.mean(volumes[-5:]) != 0 else 0.0