#!/usr/bin/env python3
"""
STRATUM-X — Adaptive Forecasting Agent for Prediction Markets
Agent for ProphetArena Competition | Sigma Lab

Architecture: Regime-Aware Ensemble with Adaptive Decision Layer
Models: LightGBM + Prophet + MLP + ARIMA + RF + XGBoost
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import logging

@dataclass
class AgentState:
    position: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_update: datetime = field(default_factory=datetime.utcnow)
    regime_streak: int = 0
    confidence_history: List[float] = field(default_factory=list)
    decision_history: List = field(default_factory=list)

class STRATUMX:
    """
    Adaptive Forecasting Agent for Prediction Markets.
    Designed for Professor Haifeng's Sigma Lab competition.

    Key capabilities:
    - Regime-aware forecasting (normal/volatile/trending/anomalous)
    - Multi-model ensemble (LightGBM + Prophet + MLP + ARIMA + RF)
    - Uncertainty-aware decision making
    - Full explainability with reasoning chains
    - Adaptive position sizing based on confidence
    """

    VERSION = "0.1.0"

    def __init__(self, symbol: str = "BTC/USDT", initial_capital: float = 10000.0):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(f"STRATUMX.{symbol}")

        self._feature_engine = None
        self._regime_detector = None
        self._ensemble = None
        self._decision_engine = None
        self._reasoning = None

        self.state = AgentState()

        self.logger.info(f"STRATUM-X initialized for {symbol}")

    def diagnose(self) -> Dict:
        return {
            "version": self.VERSION,
            "symbol": self.symbol,
            "state": {
                "position": self.state.position,
                "entry_price": self.state.entry_price,
                "regime_streak": self.state.regime_streak,
                "decisions_made": len(self.state.decision_history)
            },
            "subsystems": {
                "feature_engine": self._feature_engine is not None,
                "regime_detector": self._regime_detector is not None,
                "ensemble": self._ensemble is not None,
                "decision_engine": self._decision_engine is not None,
                "reasoning": self._reasoning is not None
            },
            "ready": all([self._feature_engine, self._regime_detector, self._ensemble, self._decision_engine])
        }

    def update(self, market_data) -> Dict:
        """Process market data and return decision."""
        import pandas as pd

        if self._feature_engine is None:
            from src.features.feature_engine import FeatureEngine
            self._feature_engine = FeatureEngine()

        if self._regime_detector is None:
            from src.regimes.regime_detector import RegimeDetector
            self._regime_detector = RegimeDetector()

        if self._ensemble is None:
            from src.models.ensemble import EnsembleForecaster
            self._ensemble = EnsembleForecaster()

        if self._decision_engine is None:
            from src.decisions.decision_engine import AdaptiveDecisionEngine
            self._decision_engine = AdaptiveDecisionEngine()

        if self._reasoning is None:
            from src.explainability.reasoning_engine import ReasoningEngine
            self._reasoning = ReasoningEngine()

        # Build features
        df = market_data if isinstance(market_data, pd.DataFrame) else market_data
        features = self._feature_engine.build(df)

        # Detect regime
        context = self._regime_detector.classify(df, features)

        # Generate forecast
        forecast = self._ensemble.predict(features, context)

        # Make decision
        decision = self._decision_engine.decide(forecast, context, self.state)

        # Generate reasoning
        reasoning = self._reasoning.generate_chain(forecast, decision, context)

        # Update state
        self.state.last_update = datetime.utcnow()
        self.state.confidence_history.append(decision.confidence)
        if context.regime != "normal":
            self.state.regime_streak += 1
        else:
            self.state.regime_streak = 0
        if self.state.position != 0:
            self.state.unrealized_pnl = (context.current_price - self.state.entry_price) * self.state.position
        self.state.decision_history.append(decision)

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": self.symbol,
            "regime": context.regime,
            "regime_confidence": context.regime_confidence,
            "volatility_zscore": context.volatility_zscore,
            "trend_strength": context.trend_strength,
            "volume_ratio": context.volume_ratio,
            "current_price": context.current_price,
            "forecast": {
                "mean": forecast.mean,
                "band_68": [forecast.lower_68, forecast.upper_68],
                "band_95": [forecast.lower_95, forecast.upper_95],
                "confidence": forecast.confidence,
                "direction": forecast.direction,
                "model_weights": forecast.model_weights
            },
            "decision": {
                "action": decision.action,
                "size": decision.size,
                "confidence": decision.confidence,
                "risk_level": decision.risk_level,
                "reasoning": decision.reasoning,
                "exit_conditions": decision.exit_conditions
            },
            "reasoning_chain": reasoning,
            "state": {
                "position": self.state.position,
                "entry_price": self.state.entry_price,
                "unrealized_pnl": self.state.unrealized_pnl,
                "realized_pnl": self.state.realized_pnl
            }
        }

    @staticmethod
    def demo():
        """Run demonstration with synthetic benchmark data."""
        print("\n" + "="*60)
        print("STRATUM-X — Adaptive Forecasting Agent")
        print("Prediction Markets | ProphetArena | Sigma Lab Competition")
        print("="*60 + "\n")

        import sys
        sys.path.insert(0, ".")
        from src.data.data_fetcher import DataFetcher

        fetcher = DataFetcher(symbol="BTCUSDT")
        agent = STRATUMX(symbol="BTC/USDT")

        scenarios = [
            ("STEADY_NORMAL", 50000, 0.015, 0.00005, ["normal"] * 400),
            ("BULL_TREND", 48000, 0.02, 0.0003, ["normal"] * 100 + ["trending"] * 300),
            ("VOLATILE_CRASH", 52000, 0.05, -0.0004, ["normal"] * 80 + ["volatile"] * 200 + ["anomalous"] * 120),
            ("WHIPSaw_CHOP", 50000, 0.025, 0.0, ["volatile"] * 400),
        ]

        results = []
        for name, base_price, vol, trend, sequence in scenarios:
            print(f"\n{'─'*60}")
            print(f"  SCENARIO: {name}")
            print(f"{'─'*60}")

            df = fetcher.generate_synthetic(
                n_periods=400, base_price=base_price,
                volatility=vol, trend=trend, regime_sequence=sequence
            )

            result = agent.update(df)

            print(f"  Regime: {result['regime'].upper()} ({result['regime_confidence']:.0%})")
            print(f"  Direction: {result['forecast']['direction']}")
            print(f"  Confidence: {result['forecast']['confidence']:.0%}")
            print(f"  Action: {result['decision']['action'].upper()}")
            print(f"  Position: {result['state']['position']}")
            print(f"  Risk: {result['decision']['risk_level'].upper()}")
            print(f"  Reasoning layers: {len(result['reasoning_chain'])}")

            results.append(result)

        print("\n" + "="*60)
        print("FINAL PERFORMANCE SUMMARY")
        print("="*60)

        regimes = {}
        actions = {}
        for r in results:
            regimes[r["regime"]] = regimes.get(r["regime"], 0) + 1
            actions[r["decision"]["action"]] = actions.get(r["decision"]["action"], 0) + 1

        avg_conf = sum(r["forecast"]["confidence"] for r in results) / len(results)
        avg_signal = sum(abs(r["forecast"]["mean"]) for r in results) / len(results)

        print(f"  Scenarios: {len(results)}")
        print(f"  Avg Confidence: {avg_conf:.1%}")
        print(f"  Avg Signal: {avg_signal:.6f}")
        print(f"  Regimes: {regimes}")
        print(f"  Actions: {actions}")
        print("\n" + "="*60 + "\n")

        return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    STRATUMX.demo()