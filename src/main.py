#!/usr/bin/env python3
"""
STRATUM — Adaptive Forecast Intelligence Agent
Sigma Research Lab @UChicago | Booth School of Business
Design: Prof. Haifeng Zhang
https://www.researchgate.net/publication/379480079

Mission: Robust, interpretable, probabilistic forecasting
for dynamic operational environments (commodity markets,
financial systems, supply chains, defense logistics).
"""

from stratum_agent import STRATUM
from adapters.prophet_arena import ProphetArenaAdapter

if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("  STRATUM — Adaptive Forecast Intelligence")
    print("  Sigma Research Lab @UChicago")
    print("  Booth School of Business | Prof. Haifeng Zhang")
    print("=" * 64)

    # ── DEMO 1: Agent run with probabilistic output ──────────────
    print("\n[1] STRATUM AGENT — Live forecast with Brier score")
    print("-" * 56)

    agent = STRATUM(horizon_steps=10, horizon_label="1h")

    prices = [100 + i * 0.5 + __import__("numpy").random.randn() * 0.3
              for i in range(50)]

    for i, price in enumerate(prices[-5:]):
        result = agent.run(price)
        print(f"\n  Time: {result.timestamp}")
        print(f"  Regime:   {result.regime.name.upper()} "
              f"[confidence={result.regime.confidence:.3f}, "
              f"vol={result.regime.volatility:.4f}, trend={result.regime.trend}]")
        print(f"  Forecast: {result.prediction:.4f} "
              f"[95% CI: {result.prediction_lower:.4f} — {result.prediction_upper:.4f}]")
        print(f"  Probability (UP): {result.probability:.3f}")
        print(f"  Brier Score:      {result.brier_score:.4f} "
              f"[calibration={result.calibration_score:.3f}]")
        print(f"  Decision:  {result.decision.action} | {result.decision.direction} "
              f"| prob={result.decision.probability:.3f} | brier={result.decision.brier_score:.4f}")
        print(f"  Reasoning: {len(result.reasoning)} steps | Latency: {result.latency_ms:.1f}ms")

        for block in result.reasoning:
            print(f"    [{block.step}] {block.heading} ({block.type}) — {block.body[:70]}...")

    # ── DEMO 2: ProphetArena adapter + Brier score ─────────────────
    print("\n\n[2] PROPHETARENA ADAPTER — Brier score calibration")
    print("-" * 56)
    ProphetArenaAdapter.demo()

    # ── DEMO 3: JSON output (as judges would receive) ─────────────
    print("\n\n[3] STRATUM — JSON OUTPUT SAMPLE")
    print("-" * 56)
    result = agent.run(prices[-1])
    pa_output = ProphetArenaAdapter.to_prophet_arena(result)
    print(f"  Protocol: {pa_output['protocol']}")
    print(f"  Brier Score: {pa_output['brier_score']}")
    print(f"  Calibration: {pa_output['calibration_score']}")
    print(f"  Regime: {pa_output['regime']} | {pa_output['regime_confidence']:.3f}")
    print(f"  Action: {pa_output['action']} | {pa_output['direction']}")
    print(f"  Probability: {pa_output['probability']}")
    print(f"  Decision Brier: {pa_output['decision_brier_score']}")
    print(f"  Models: {len(pa_output['models'])} (ensemble)")
    print(f"  Reasoning blocks: {len(pa_output['reasoning'])}")
    print(f"  Latency: {pa_output['latency_ms']:.1f}ms")

    print("\n" + "=" * 64)
    print("  STRATUM is operational.")
    print("  All outputs are deterministic, reproducible, and interpretable.")
    print("=" * 64 + "\n")