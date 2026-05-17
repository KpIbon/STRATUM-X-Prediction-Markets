# STRATUM-X
## Adaptive Forecasting Agent for Prediction Markets

**ProphetArena Competition | Sigma Lab | Professor Haifeng**

---

## Judging Rubric

| Criterion | Weight | STRATUM-X Approach |
|-----------|--------|-------------------|
| **Creativity** | 30% | Regime-aware ensemble with adaptive thresholds, unique forecasting-as-systems-design framing |
| **Implementation** | 40% | Clean Python, 8-module architecture, full dataclass typing, ProphetArena SDK adapter |
| **Robustness** | 30% | Tested across 8 market scenarios, confidence scoring, regime override logic |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run demo
python run.py

# Export submission package
python -m zipfile -c stratum_x_submission.zip src/ run.py requirements.txt README_SUBMISSION.md
```

---

## Architecture

```
src/
├── agents/stratum_agent.py       ← Main agent (STRATUMX class)
├── data/data_fetcher.py           ← Binance + synthetic data
├── features/feature_engine.py     ← 50+ engineered features
├── regimes/regime_detector.py      ← 4-regime online classifier
├── models/ensemble.py             ← LightGBM + Prophet + MLP + ARIMA + RF
├── decisions/decision_engine.py    ← Kelly-inspired adaptive sizing
├── explainability/reasoning_engine.py ← 7-layer explainability
└── api/prophet_arena.py          ← ProphetArena competition adapter
```

---

## 8 Test Scenarios

1. **STEADY_NORMAL** — Baseline stable market
2. **BULL_TREND** — Persistent upward trend with trending regime
3. **VOLATILE_CRASH** — High volatility with anomalous regime
4. **RANGE_BOUND** — Sideways market
5. **WHIPSaw_CHOP** — Volatile with no direction
6. **TREND_REVERSAL** — Trending → volatile → normal transitions
7. **LIQUIDITY_CRISIS** — Anomalous + volatile combined
8. **SLOW_ACCUMULATION** — Long-term normal with slight drift

---

## Competition Features

- **4 Regimes**: normal, volatile, trending, anomalous
- **5 Models**: LightGBM, Prophet, MLP, ARIMA, Random Forest
- **7 Explainability Layers**: Executive summary → Forecast → Regime → Decision → Architecture → Risk → Competition readiness
- **Regime-Adaptive Thresholds**: buy/sell/min-confidence/stop-loss/take-profit all adjust per regime
- **Kelly-Inspired Position Sizing**: confidence-weighted with conservative cap
- **ProphetArena SDK**: adapter implements ai-prophet-core interface

---

## Run Command (for ProphetArena Evaluation)

```bash
python run.py
```

## Export for Submission

```bash
python -m zipfile -c stratum_x_submission.zip src/ run.py requirements.txt README_SUBMISSION.md
```

Or use the included export script:

```bash
bash scripts/export.sh
```