"""
Adaptive Policy — maps (forecast, regime, confidence) → action.
Actions: buy | sell | hold | abstain
Thresholds dynamically adjust per regime to maximize Sharpe in each state.
"""

import logging
from typing import Dict, List

log = logging.getLogger("stratum.decision")


class AdaptivePolicy:
    """
    Threshold-based policy that adapts to regime.
    Buy/sell thresholds widen in volatile regimes to reduce false signals.
    """

    # Regime-conditional thresholds
    REGIME_PARAMS = {
        "stable":    {"buy_thresh": 0.55, "sell_thresh": 0.45, "conf_min": 0.65, "position": 1.0},
        "volatile":  {"buy_thresh": 0.65, "sell_thresh": 0.35, "conf_min": 0.70, "position": 0.5},
        "transition":{"buy_thresh": 0.60, "sell_thresh": 0.40, "conf_min": 0.75, "position": 0.4},
        "anomalous": {"buy_thresh": 0.75, "sell_thresh": 0.25, "conf_min": 0.80, "position": 0.0},
    }

    def compute(self, forecast: Dict, regime: Dict) -> Dict:
        """
        Returns: {action, confidence, alternatives, rationale}
        """
        value = forecast.get("value", 0.5)
        conf = forecast.get("confidence", 0.5)
        regime_state = regime.get("state", "stable")

        params = self.REGIME_PARAMS.get(regime_state, self.REGIME_PARAMS["stable"])
        buy_thresh = params["buy_thresh"]
        sell_thresh = params["sell_thresh"]
        conf_min = params["conf_min"]
        position_scale = params["position"]

        # Adjust threshold by confidence
        conf_factor = max(0.5, min(1.0, conf / conf_min))
        adjusted_buy = buy_thresh + (0.5 - buy_thresh) * (1 - conf_factor) * 0.3
        adjusted_sell = sell_thresh + (0.5 - sell_thresh) * (1 - conf_factor) * 0.3

        if value > adjusted_buy:
            action = "buy"
        elif value < adjusted_sell:
            action = "sell"
        else:
            action = "hold"

        # Compress confidence when regime is uncertain
        regime_conf = regime.get("confidence", 0.8)
        adjusted_conf = conf * regime_conf

        if adjusted_conf < conf_min:
            action = "abstain" if action in ("buy", "sell") else action

        rationale = self._build_rationale(
            action, value, conf, regime_state, adjusted_buy, adjusted_sell
        )

        # Scored alternatives
        alternatives = self._score_alternatives(action, value, adjusted_buy, adjusted_sell, conf)

        return {
            "action": action,
            "confidence": round(float(adjusted_conf), 3),
            "alternatives": alternatives,
            "rationale": rationale,
            "regime_used": regime_state,
            "thresholds": {
                "buy": round(adjusted_buy, 3),
                "sell": round(adjusted_sell, 3),
                "conf_min": conf_min,
            },
            "position_scale": position_scale,
        }

    @staticmethod
    def _build_rationale(
        action: str, value: float, conf: float,
        regime: str, buy_t: float, sell_t: float
    ) -> str:
        conf_pct = f"{conf * 100:.0f}%"
        val_pct = f"{value * 100:.1f}%"
        if action == "buy":
            return (f"[{regime.upper()}] Forecast {val_pct} exceeds buy threshold "
                    f"{buy_t:.2f} at {conf_pct} confidence. Converging signals → BUY.")
        elif action == "sell":
            return (f"[{regime.upper()}] Forecast {val_pct} below sell threshold "
                    f"{sell_t:.2f} at {conf_pct} confidence. Diverging signals → SELL.")
        elif action == "hold":
            return (f"[{regime.upper()}] Forecast {val_pct} within [{sell_t:.2f}, {buy_t:.2f}] "
                    f"range. Neutral signal at {conf_pct} confidence → HOLD.")
        else:
            return (f"[{regime.upper()}] Confidence {conf_pct} below minimum for "
                    f"position-taking. Elevated uncertainty → ABSTAIN.")

    @staticmethod
    def _score_alternatives(
        action: str, value: float, buy_t: float, sell_t: float, conf: float
    ) -> List[Dict]:
        base = {"confidence": round(conf, 3), "score": 0.0}
        alternatives = []

        for a in ["buy", "sell", "hold"]:
            score = 0.5
            if a == action:
                score = conf
            elif a == "buy":
                score = max(0, (value - buy_t) / (1 - buy_t)) if value > buy_t else 0.0
            elif a == "sell":
                score = max(0, (sell_t - value) / sell_t) if value < sell_t else 0.0
            alternatives.append({**base, "action": a, "score": round(score, 3)})

        return sorted(alternatives, key=lambda x: x["score"], reverse=True)