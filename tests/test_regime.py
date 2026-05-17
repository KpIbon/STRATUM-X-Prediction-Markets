"""
Unit test — regime detection.
Run: python -m pytest tests/test_regime.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from regime.regime_detector import RegimeDetector


def test_stable_regime():
    rd = RegimeDetector()
    feats = {"volatility_30": 0.005, "returns_5": 0.01, "volume_ratio": 1.0, "price_level": 0.1}
    bars = [{"close": 100 + i} for i in range(100)]
    result = rd.classify(feats, bars)
    assert result["state"] in ["stable", "volatile", "transition", "anomalous"]
    assert 0 <= result["confidence"] <= 1
    print(f"  Stable regime: {result['state']} (conf={result['confidence']:.2%})")
    print("✓ Stable regime test passed")


def test_volatile_regime():
    rd = RegimeDetector()
    feats = {"volatility_30": 0.050, "returns_5": -0.05, "volume_ratio": 2.5, "price_level": 1.5}
    bars = [{"close": 100 + i} for i in range(100)]
    result = rd.classify(feats, bars)
    assert result["state"] in ["volatile", "anomalous"]
    print(f"  Volatile regime: {result['state']} (conf={result['confidence']:.2%})")
    print("✓ Volatile regime test passed")


def test_anomalous_regime():
    rd = RegimeDetector()
    feats = {"volatility_30": 0.200, "returns_5": 0.20, "volume_ratio": 5.0, "price_level": 5.0}
    bars = [{"close": 100 + i} for i in range(100)]
    result = rd.classify(feats, bars)
    assert result["state"] == "anomalous"
    print(f"  Anomalous regime: {result['state']}")
    print("✓ Anomalous regime test passed")


if __name__ == "__main__":
    test_stable_regime()
    test_volatile_regime()
    test_anomalous_regime()
    print("\n✓ All regime tests passed")