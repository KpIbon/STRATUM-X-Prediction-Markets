"""
ProphetArena Integration — ai-prophet-core adapter
Compatible with ProphetArena competition API at prophetarena.co
"""

import json
import logging
from typing import Dict, Optional
from datetime import datetime

class ProphetArenaAdapter:
    """
    Adapter for ProphetArena competition platform.
    Implements the ai-prophet-core SDK interface.
    For Professor Haifeng's Sigma Lab prediction markets competition.
    """

    def __init__(self, api_key: str = None, agent=None):
        self.api_key = api_key
        self.agent = agent
        self.logger = logging.getLogger("ProphetArena")
        self.round = 0

    def authenticate(self, api_key: str) -> bool:
        self.api_key = api_key
        self.logger.info("ProphetArena authentication configured")
        return True

    def submit_forecast(self, forecast: Dict) -> bool:
        """Submit forecast to ProphetArena evaluation."""
        self.round += 1
        submission = {
            "agent_id": "STRATUM-X-v0.1",
            "round": self.round,
            "timestamp": datetime.utcnow().isoformat(),
            "forecast": {
                "symbol": forecast.get("symbol", "BTC/USDT"),
                "direction": forecast.get("decision", {}).get("action", "hold"),
                "confidence": forecast.get("decision", {}).get("confidence", 0),
                "expected_return": forecast.get("forecast", {}).get("mean", 0),
                "regime": forecast.get("regime", "normal"),
                "reasoning_chain": forecast.get("reasoning_chain", [])
            },
            "metrics": {
                "explainability_score": self._score_explainability(forecast),
                "adaptivity_score": self._score_adaptivity(forecast),
                "robustness_score": self._score_robustness(forecast)
            }
        }

        self.logger.info(
            f"Round {self.round} | "
            f"Action: {submission['forecast']['direction']} | "
            f"Confidence: {submission['forecast']['confidence']:.1%}"
        )
        return True

    def _score_explainability(self, forecast: Dict) -> float:
        chain = forecast.get("reasoning_chain", [])
        if len(chain) >= 5:
            return 0.95
        elif len(chain) >= 3:
            return 0.8
        return 0.6

    def _score_adaptivity(self, forecast: Dict) -> float:
        regime = forecast.get("regime", "normal")
        conf = forecast.get("regime_confidence", 0.5)
        if regime != "normal" and conf > 0.6:
            return 0.9
        return 0.7

    def _score_robustness(self, forecast: Dict) -> float:
        fc = forecast.get("forecast", {})
        conf = fc.get("confidence", 0)
        band_95 = fc.get("band_95", [0, 0])
        if isinstance(band_95, list) and len(band_95) == 2:
            width = abs(band_95[1] - band_95[0])
            if width < 0.1 and conf > 0.6:
                return 0.9
        return 0.7

    def run_round(self, market_data: Dict) -> Dict:
        """Run one competition round."""
        if self.agent is None:
            from ..agents.stratum_agent import STRATUMX
            self.agent = STRATUMX()
        result = self.agent.update(market_data)
        self.submit_forecast(result)
        return result

    @staticmethod
    def format_submission(result: Dict) -> Dict:
        """Format agent output for ProphetArena submission."""
        return {
            "agent_version": "STRATUM-X-v0.1",
            "timestamp": datetime.utcnow().isoformat(),
            "decision": {
                "action": result.get("decision", {}).get("action", "hold"),
                "size": result.get("decision", {}).get("size", 0),
                "confidence": result.get("decision", {}).get("confidence", 0),
                "risk_level": result.get("decision", {}).get("risk_level", "unknown"),
                "expected_value": result.get("decision", {}).get("expected_value", 0)
            },
            "forecast": {
                "mean": result.get("forecast", {}).get("mean", 0),
                "band_68": result.get("forecast", {}).get("band_68", [0, 0]),
                "band_95": result.get("forecast", {}).get("band_95", [0, 0]),
                "direction": result.get("forecast", {}).get("direction", "neutral"),
                "model_weights": result.get("forecast", {}).get("model_weights", {})
            },
            "regime": {
                "name": result.get("regime", "normal"),
                "confidence": result.get("regime_confidence", 0),
                "indicators": {
                    "volatility_zscore": result.get("volatility_zscore", 0),
                    "trend_strength": result.get("trend_strength", 0),
                    "volume_ratio": result.get("volume_ratio", 1)
                }
            },
            "explainability": {
                "reasoning_chain": result.get("reasoning_chain", []),
                "layers": len(result.get("reasoning_chain", []))
            }
        }