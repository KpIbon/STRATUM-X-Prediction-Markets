"""
Adaptive Decision Engine — maps forecasts to actionable decisions
"""

from dataclasses import dataclass, field
from typing import List, Dict
import numpy as np

@dataclass
class Decision:
    action: str
    size: float
    confidence: float
    reasoning: List[str]
    risk_level: str
    expected_value: float
    regime_override: bool
    exit_conditions: List[str]

@dataclass
class MarketContext:
    symbol: str = "BTC/USDT"
    current_price: float = 0.0
    timestamp: object = None
    regime: str = "normal"
    regime_confidence: float = 0.5
    volatility_zscore: float = 0.0
    trend_strength: float = 0.0
    volume_ratio: float = 1.0
    price_momentum: float = 0.0
    regime_history: List = field(default_factory=list)
    price_level: float = 0.0

@dataclass
class AgentState:
    position: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_update: object = None
    regime_streak: int = 0
    confidence_history: List = field(default_factory=list)
    decision_history: List = field(default_factory=list)

@dataclass
class ForecastResult:
    horizon: int = 15
    mean: float = 0.0
    lower_68: float = -0.02
    upper_68: float = 0.02
    lower_95: float = -0.04
    upper_95: float = 0.04
    confidence: float = 0.3
    regime: str = "normal"
    model_weights: Dict = field(default_factory=dict)
    feature_importance: Dict = field(default_factory=dict)
    direction: str = "neutral"

class AdaptiveDecisionEngine:
    """
    Converts regime + forecast into position decisions.
    Features:
    - Regime-adaptive thresholds
    - Kelly-inspired position sizing
    - Confidence-weighted sizing
    - Exit condition automation
    """

    def decide(self, forecast: ForecastResult, context: MarketContext, state: AgentState) -> Decision:
        """Main decision method — returns actionable Decision."""
        thresholds = self._get_thresholds(context.regime)
        reasoning = []

        signal = forecast.mean
        confidence = forecast.confidence
        direction = forecast.direction

        regime_override = False
        if context.regime == "anomalous" and confidence < 0.6:
            regime_override = True
            reasoning.append(f"REGIME OVERRIDE: {context.regime} with low confidence ({confidence:.1%})")
        if context.regime == "volatile" and state.regime_streak > 5:
            regime_override = True
            reasoning.append(f"REGIME OVERRIDE: Extended volatile streak ({state.regime_streak} rounds)")

        if regime_override:
            action, risk = self._override_action(context, state, reasoning)
            size = 0.0
        else:
            action, risk = self._normal_action(direction, signal, confidence, thresholds, reasoning)
            size = self._compute_position_size(action, confidence, context.regime, thresholds, state)

        exit_conditions = self._generate_exit_conditions(action, context)
        ev = self._compute_expected_value(action, forecast, confidence)
        reasoning.append(f"Direction: {direction.upper()} | Signal: {signal:.4%} | Confidence: {confidence:.1%} | Regime: {context.regime.upper()}")

        return Decision(
            action=action,
            size=round(size, 4),
            confidence=round(confidence, 3),
            reasoning=reasoning,
            risk_level=risk,
            expected_value=round(ev, 4),
            regime_override=regime_override,
            exit_conditions=exit_conditions
        )

    def _get_thresholds(self, regime: str) -> Dict[str, float]:
        base = {
            "min_confidence_buy": 0.55,
            "min_confidence_sell": 0.52,
            "signal_threshold_buy": 0.002,
            "signal_threshold_sell": -0.002,
            "max_position": 1.0,
            "stop_loss": 0.02,
            "take_profit": 0.04
        }
        overrides = {
            "normal": {},
            "trending": {"min_confidence_buy": 0.50, "min_confidence_sell": 0.48, "max_position": 1.25, "stop_loss": 0.03, "take_profit": 0.06},
            "volatile": {"min_confidence_buy": 0.65, "min_confidence_sell": 0.62, "max_position": 0.6, "stop_loss": 0.015, "take_profit": 0.03},
            "anomalous": {"min_confidence_buy": 0.75, "min_confidence_sell": 0.72, "max_position": 0.3, "stop_loss": 0.01, "take_profit": 0.02}
        }
        return {**base, **overrides.get(regime, {})}

    def _normal_action(self, direction: str, signal: float, confidence: float, thresholds: Dict, reasoning: List[str]) -> tuple:
        risk = "medium"
        if direction == "up" and signal > thresholds["signal_threshold_buy"] and confidence >= thresholds["min_confidence_buy"]:
            action = "buy"
            reasoning.append(f"BUY SIGNAL: {direction}, signal={signal:.4%}, conf={confidence:.1%}")
        elif direction == "down" and signal < thresholds["signal_threshold_sell"] and confidence >= thresholds["min_confidence_sell"]:
            action = "sell"
            reasoning.append(f"SELL SIGNAL: {direction}, signal={signal:.4%}, conf={confidence:.1%}")
        elif abs(signal) < 0.001:
            action = "hold"
            risk = "low"
            reasoning.append(f"HOLD: Signal too weak ({signal:.4%})")
        else:
            action = "reduce"
            risk = "medium"
            reasoning.append("REDUCE: Mixed signals")
        return action, risk

    def _override_action(self, context: MarketContext, state: AgentState, reasoning: List[str]) -> tuple:
        if state.position > 0:
            reasoning.append("OVERRIDE: Closing long position due to adverse regime")
            return "sell", "high"
        else:
            reasoning.append("OVERRIDE: No entry during high-risk regime")
            return "hold", "high"

    def _compute_position_size(self, action: str, confidence: float, regime: str, thresholds: Dict, state: AgentState) -> float:
        if action in ["hold"]:
            return 0.0
        kelly = confidence * 0.3
        mult = {"normal": 1.0, "trending": 1.2, "volatile": 0.6, "anomalous": 0.3}
        size = min(kelly * mult.get(regime, 1.0), thresholds["max_position"])
        if state.position > 0 and action == "buy":
            size = min(size, 0.3)
        elif state.position > 0 and action == "sell":
            size = 0.5
        return max(0.0, size)

    def _generate_exit_conditions(self, action: str, context: MarketContext) -> List[str]:
        conditions = []
        if action in ["buy", "sell"]:
            sl = 0.015 if context.regime == "volatile" else 0.02
            tp = 0.03 if context.regime == "volatile" else 0.04
            conditions.append(f"Stop loss: {sl:.1%}")
            conditions.append(f"Take profit: {tp:.1%}")
            conditions.append("Exit if regime shifts to anomalous")
        if context.regime == "volatile":
            conditions.append("Hard exit if drawdown exceeds 2%")
            conditions.append("Reduce to 50% if confidence drops below 0.5")
        return conditions

    def _compute_expected_value(self, action: str, forecast: ForecastResult, confidence: float) -> float:
        if action == "hold":
            return 0.0
        signal = forecast.mean
        spread = (forecast.upper_68 - forecast.lower_68) / 2
        if action == "buy":
            return signal * confidence - spread * (1 - confidence)
        elif action == "sell":
            return -signal * confidence - spread * (1 - confidence)
        return 0.0