"""
Benchmark Runner — runs STRATUM-X through synthetic + live market scenarios
Produces performance reports for ProphetArena submission
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict
import sys

class BenchmarkRunner:
    """
    Runs the agent through multiple market scenarios.
    Generates performance metrics for competition judging.
    """

    def __init__(self, symbol: str = "BTC/USDT", n_rounds: int = 10):
        self.symbol = symbol
        self.n_rounds = n_rounds
        self.results: List[Dict] = []

    def run(self) -> List[Dict]:
        """Run benchmark across multiple market scenarios."""
        from ..data.data_fetcher import DataFetcher
        from ..agents.stratum_agent import STRATUMX

        fetcher = DataFetcher(symbol=self.symbol.replace("/", ""))
        agent = STRATUMX(symbol=self.symbol)

        scenarios = self._define_scenarios()

        for i, scenario in enumerate(scenarios[:self.n_rounds]):
            print(f"\n{'─'*60}")
            print(f"  SCENARIO {i+1}/{min(self.n_rounds, len(scenarios))}: {scenario['name']}")
            print(f"{'─'*60}")

            # Generate market data for scenario
            df = fetcher.generate_synthetic(
                n_periods=scenario.get("n_periods", 500),
                base_price=scenario.get("base_price", 50000),
                volatility=scenario.get("volatility", 0.02),
                trend=scenario.get("trend", 0.0001),
                regime_sequence=scenario.get("regime_sequence", None)
            )

            # Run agent
            result = agent.update(df)

            # Print summary
            print(f"  Regime:        {result['regime'].upper()} ({result['regime_confidence']:.0%})")
            print(f"  Direction:    {result['forecast']['direction']}")
            print(f"  Confidence:   {result['forecast']['confidence']:.0%}")
            print(f"  Signal:       {result['forecast']['mean']:.6f}")
            print(f"  Action:       {result['decision']['action'].upper()}")
            print(f"  Position:     {result['state']['position']}")
            print(f"  P&L:          ${result['state']['unrealized_pnl']:.2f}")
            print(f"  Risk Level:   {result['decision']['risk_level'].upper()}")

            self.results.append(result)

        return self.results

    def _define_scenarios(self) -> List[Dict]:
        """Define benchmark scenarios — each tests different market conditions."""
        return [
            {
                "name": "STEADY_NORMAL",
                "base_price": 50000,
                "volatility": 0.015,
                "trend": 0.00005,
                "n_periods": 500,
                "regime_sequence": ["normal"] * 500
            },
            {
                "name": "BULL_TREND",
                "base_price": 48000,
                "volatility": 0.02,
                "trend": 0.0003,
                "n_periods": 500,
                "regime_sequence": ["normal"] * 100 + ["trending"] * 400
            },
            {
                "name": "VOLATILE_CRASH",
                "base_price": 52000,
                "volatility": 0.05,
                "trend": -0.0004,
                "n_periods": 400,
                "regime_sequence": ["normal"] * 100 + ["volatile"] * 200 + ["anomalous"] * 100
            },
            {
                "name": "RANGE_BOUND",
                "base_price": 50000,
                "volatility": 0.01,
                "trend": 0.0,
                "n_periods": 500,
                "regime_sequence": ["normal"] * 500
            },
            {
                "name": "WHIPSaw_CHOP",
                "base_price": 50000,
                "volatility": 0.025,
                "trend": 0.0,
                "n_periods": 400,
                "regime_sequence": ["volatile"] * 400
            },
            {
                "name": "TREND_REVERSAL",
                "base_price": 45000,
                "volatility": 0.02,
                "trend": 0.0002,
                "n_periods": 500,
                "regime_sequence": ["trending"] * 200 + ["volatile"] * 150 + ["normal"] * 150
            },
            {
                "name": "LIQUIDITY_CRISIS",
                "base_price": 51000,
                "volatility": 0.06,
                "trend": -0.0002,
                "n_periods": 300,
                "regime_sequence": ["normal"] * 50 + ["volatile"] * 100 + ["anomalous"] * 100 + ["volatile"] * 50
            },
            {
                "name": "SLOW_ACCUMULATION",
                "base_price": 47000,
                "volatility": 0.012,
                "trend": 0.00015,
                "n_periods": 600,
                "regime_sequence": ["normal"] * 600
            }
        ]

    def generate_report(self) -> Dict:
        """Generate a comprehensive benchmark report."""
        if not self.results:
            return {}

        all_decisions = [r["decision"] for r in self.results]
        all_regimes = [r["regime"] for r in self.results]

        regime_counts = {}
        action_counts = {}
        for r in self.results:
            regime_counts[r["regime"]] = regime_counts.get(r["regime"], 0) + 1
            action_counts[r["decision"]["action"]] = action_counts.get(r["decision"]["action"], 0) + 1

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_rounds": len(self.results),
            "regime_distribution": regime_counts,
            "action_distribution": action_counts,
            "avg_confidence": np.mean([r["forecast"]["confidence"] for r in self.results]),
            "avg_signal_strength": np.mean([abs(r["forecast"]["mean"]) for r in self.results]),
            "regime_override_count": sum(1 for d in all_decisions if d.get("regime_override")),
            "final_capital": self.results[-1]["state"].get("position", 0) * self.results[-1].get("current_price", 0) if self.results else 0,
            "total_realized_pnl": sum(r["state"].get("realized_pnl", 0) for r in self.results),
            "worst_regime": max(regime_counts, key=regime_counts.get),
            "most_common_action": max(action_counts, key=action_counts.get)
        }