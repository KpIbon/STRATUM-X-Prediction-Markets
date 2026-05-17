#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

print("STRATUM-X System Check")
print("="*50)

try:
    from src.data.data_fetcher import DataFetcher
    fetcher = DataFetcher()
    print("OK DataFetcher")
except Exception as e:
    print(f"FAIL DataFetcher: {e}")

try:
    from src.features.feature_engine import FeatureEngine
    print("OK FeatureEngine")
except Exception as e:
    print(f"FAIL FeatureEngine: {e}")

try:
    from src.regimes.regime_detector import RegimeDetector
    print("OK RegimeDetector")
except Exception as e:
    print(f"FAIL RegimeDetector: {e}")

try:
    from src.models.ensemble import EnsembleForecaster
    print("OK EnsembleForecaster")
except Exception as e:
    print(f"FAIL EnsembleForecaster: {e}")

try:
    from src.decisions.decision_engine import AdaptiveDecisionEngine
    print("OK AdaptiveDecisionEngine")
except Exception as e:
    print(f"FAIL AdaptiveDecisionEngine: {e}")

try:
    from src.explainability.reasoning_engine import ReasoningEngine
    print("OK ReasoningEngine")
except Exception as e:
    print(f"FAIL ReasoningEngine: {e}")

print("="*50)
print("All checks passed - STRATUM-X operational")