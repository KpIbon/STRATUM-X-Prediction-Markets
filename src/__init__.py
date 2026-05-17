# ─────────────────────────────────────────────────────────────────
# STRATUM — Adaptive Forecast Intelligence for Dynamic Systems
# Sigma Lab AI Forecasting Track — ProphetArena Compatible Agent
# ─────────────────────────────────────────────────────────────────
#
# "Interpret → Forecast → Adapt → Explain"
#
# STRATUM is an adaptive operational intelligence system that reads
# live market data, detects environmental regimes, generates ensemble
# forecasts, and produces explainable trading decisions — all from
# a single command:  python main.py
#
# ─── Architecture ───────────────────────────────────────────────
#
#   data/         — Data ingestion from multiple sources (REST,
#                   WebSocket, file). Handles missing data and
#                   normalization into a unified time-series format.
#
#   features/     — Rolling-window feature engineering. Generates
#                   momentum, volatility, regime, and temporal
#                   features across multiple horizons (5, 15, 30,
#                   60 bars).
#
#   regime/       — Online regime detector. Classifies market into:
#                   stable | volatile | transition | anomalous.
#                   Uses statistical thresholds + gradient detection.
#
#   forecast/     — Ensemble forecaster. Combines RandomForest,
#                   XGBoost, and LightGBM into a stacked meta-model.
#                   Produces a single forecast with calibrated
#                   confidence intervals.
#
#   decision/     — Adaptive policy layer. Maps (regime, forecast,
#                   confidence) → action (buy / sell / hold / abstain).
#                   Thresholds adjust dynamically per regime.
#
#   explain/      — Reasoning engine. Generates a structured 4-step
#                   reasoning chain (Observation → Analysis →
#                   Decision → Alert) for every output.
#
#   adapters/     — ProphetArena compatibility layer. Converts
#                   inputs/outputs to/from the standardized format
#                   used by the evaluation framework.
#
#   models/       — Individual model implementations:
#                   random_forest.py, xgboost_model.py,
#                   lightgbm_model.py, ensemble_stack.py
#
#   tests/        — Unit + integration tests for each module.
#
# ─── Execution Modes ───────────────────────────────────────────
#
#   python main.py                              # single symbol (default)
#   python main.py --mode batch --input ./data/  # batch benchmark
#   python main.py --symbol ETH-USD --lookback 500
#   python main.py --output result.json --verbose
#
# ─── Model Stack ───────────────────────────────────────────────
#
#   1. RandomForest (base learner, robust to noise)
#   2. XGBoost       (gradient boosting, captures non-linearity)
#   3. LightGBM      (fast tabular, regime-adaptive weights)
#   4. Ensemble      (meta-learner blending the three above)
#
# ─── Regime Logic ───────────────────────────────────────────────
#
#   STABLE     : low volatility, high confidence → aggressive action
#   VOLATILE   : high volatility, compressed confidence → defensive
#   TRANSITION : regime shift detected → reduce position, wait
#   ANOMALOUS  : outlier event detected → abstain, log alert
#
# ─── Confidence-Aware Decisions ───────────────────────────────
#
#   confidence > 0.80  → full conviction
#   0.60 < conf < 0.80  → reduced position
#   0.40 < conf < 0.60  → hold
#   conf < 0.40         → abstain (degraded mode)
#
# ─── Explainability ────────────────────────────────────────────
#
#   Every action includes:
#   - forecast value + confidence
#   - detected regime + regime confidence
#   - decision action + policy rationale
#   - 4-block reasoning chain (human-readable)
#   - feature attribution (top 3 contributors)
#
# ─────────────────────────────────────────────────────────────────

from data.market_data import MarketDataFetcher
from regime.regime_detector import RegimeDetector
from forecast.ensemble import EnsembleForecaster
from decision.adaptive_policy import AdaptivePolicy
from explain.reasoning import ReasoningEngine

__version__ = "1.0.0"
__author__ = "KpIbon"

__all__ = [
    "STRATUMAgent",
    "MarketDataFetcher",
    "RegimeDetector",
    "EnsembleForecaster",
    "AdaptivePolicy",
    "ReasoningEngine",
]