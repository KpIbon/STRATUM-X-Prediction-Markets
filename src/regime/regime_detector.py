"""
Regime Detection — online classification of market state.
Classifies into: stable | volatile | transition | anomalous
Uses statistical thresholds on volatility, drift, and volume spikes.
"""

import logging
from typing import Dict

import numpy as np

log = logging.getLogger("stratum.regime")


class RegimeDetector:
    """
    Rule-based regime classifier. No ML model needed —
    stability and performance come from well-tuned thresholds.
    """

    def __init__(self):
        self._history_volatility: list = []
        self._history_returns: list = []
        # Baseline std used until enough real history accumulates
        self._baseline_std = 0.005

    def classify(self, features: Dict[str, float], lookback: list) -> Dict:
        """
        Returns: {state, confidence, metrics, transitions}
        """
        vol = features.get("volatility_30", 0)
        ret = features.get("returns_5", 0)
        vol_ratio = features.get("volume_ratio", 1)
        price_level = features.get("price_level", 0)

        self._history_volatility.append(vol)
        self._history_returns.append(ret)

        # Build a rolling estimate of volatility stats
        recent_vols = self._history_volatility[-60:]
        historical_mean = np.mean(recent_vols) if recent_vols else 0.01
        if len(recent_vols) > 4:
            historical_std = max(np.std(recent_vols), self._baseline_std)
        else:
            historical_std = self._baseline_std  # not enough data yet

        vol_score = (vol - historical_mean) / historical_std

        # Anomalous — extreme vol or volume spike
        if vol_score > 3.0 or vol_ratio > 3.0 or abs(price_level) > 4:
            state = "anomalous"
        # Volatile — elevated relative to history OR intrinsically high
        elif vol_score > 1.0 or vol > 0.030:  # 0.03 = ~6x normal (rule-based backup)
            state = "volatile"
        # Transition — rapidly changing regime
        elif len(self._history_returns) > 5 and np.std(self._history_returns[-10:]) > 0.015:
            state = "transition"
        else:
            state = "stable"

        state_confidence = self._compute_confidence(vol_score, vol_ratio, state)

        log.info(f"Regime classified: {state} (confidence={state_confidence:.2%})")

        return {
            "state": state,
            "confidence": round(state_confidence, 3),
            "metrics": {
                "volatility_score": round(vol_score, 3),
                "volume_ratio": round(vol_ratio, 2),
                "price_level_z": round(price_level, 3),
                "drift_5": round(ret, 4),
                "baseline_vol": round(historical_mean, 5),
            },
            "transitions": {
                "was_stable": state != "stable",
                "volatility_up": vol_score > 1.0,
                "volume_spike": vol_ratio > 2.0,
            },
        }

    def _compute_confidence(self, vol_score: float, vol_ratio: float, state: str) -> float:
        if state == "anomalous":
            return min(0.95, 0.70 + min(abs(vol_score) * 0.05, 0.20) + min(vol_ratio * 0.05, 0.10))
        elif state == "volatile":
            return min(0.88, 0.65 + min(vol_score * 0.05, 0.15))
        elif state == "transition":
            return min(0.80, 0.55 + min(vol_score * 0.03, 0.15))
        else:
            return min(0.92, 0.75 + max(0, (1.0 - abs(vol_score)) * 0.10))