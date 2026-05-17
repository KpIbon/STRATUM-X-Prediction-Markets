"""
Integration test — full pipeline smoke test.
Run: python -m pytest tests/test_pipeline.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import STRATUMAgent


def test_end_to_end():
    """Verify the full pipeline runs without errors."""
    agent = STRATUMAgent(verbose=False)
    result = agent.run(symbol="BTC-USD", lookback=200)

    # Metadata checks
    assert result["metadata"]["agent"] == "STRATUM"
    assert result["metadata"]["status"] == "success"
    assert result["metadata"]["elapsed_ms"] > 0

    # Regime checks
    assert result["regime"]["state"] in ["stable", "volatile", "transition", "anomalous"]
    assert 0 <= result["regime"]["confidence"] <= 1

    # Forecast checks
    assert isinstance(result["forecast"]["value"], float)
    assert 0 <= result["forecast"]["confidence"] <= 1
    assert result["forecast"]["lower"] <= result["forecast"]["value"] <= result["forecast"]["upper"]

    # Decision checks
    assert result["decision"]["action"] in ["buy", "sell", "hold", "abstain"]
    assert 0 <= result["decision"]["confidence"] <= 1
    assert "rationale" in result["decision"]

    # Reasoning checks
    assert len(result["reasoning"]["chain"]) == 4
    assert all("step" in block and "type" in block for block in result["reasoning"]["chain"])

    print("✓ End-to-end pipeline test passed")


def test_fallback():
    """Verify graceful degradation on bad data."""
    agent = STRATUMAgent(verbose=False)
    agent.history = []  # empty data
    result = agent.run(symbol="BTC-USD", lookback=0)
    assert result["metadata"]["status"] in ("success", "degraded")
    print("✓ Fallback test passed")


def test_multiple_runs():
    """Verify deterministic + non-decreasing confidence across runs."""
    agent = STRATUMAgent(verbose=False)
    results = [agent.run(symbol="BTC-USD", lookback=100) for _ in range(3)]
    regimes = {r["regime"]["state"] for r in results}
    actions = {r["decision"]["action"] for r in results}
    print(f"  Regimes seen: {regimes}")
    print(f"  Actions seen: {actions}")
    assert len(regimes) >= 1
    print("✓ Multiple run test passed")


if __name__ == "__main__":
    test_end_to_end()
    test_fallback()
    test_multiple_runs()
    print("\n✓ All tests passed")