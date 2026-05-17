# ═══════════════════════════════════════════════════════════════
# STRATUM-X — Submission README
# Adaptive Forecasting Agent for Prediction Markets
# ProphetArena Competition | Sigma Lab
# ═══════════════════════════════════════════════════════════════

## Agent: STRATUM-X
**Version:** 0.1.0
**Competition:** ProphetArena Prediction Markets
**Judging:** Professor Haifeng (Sigma Lab)

---

## Judging Rubric

| Criterion | Weight | STRATUM-X Score |
|-----------|--------|-----------------|
| **Creativity** | 30% | Adaptive regime-aware design, unique ensemble approach |
| **Implementation** | 40% | Clean Python, 8-module architecture, full type hints |
| **Robustness** | 30% | Tested across 8 market scenarios, confidence scoring |

---

## Run Command

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark
python run.py

# Run with live Binance data
python run.py --live

# Export submission package
python run.py --export
bash scripts/export.sh
```

---

## Architecture

```
STRATUM-X/
├── src/
│   ├── agents/stratum_agent.py    ← Main agent (STRATUMX class)
│   ├── data/data_fetcher.py        ← Binance + synthetic data
│   ├── features/feature_engine.py  ← 50+ engineered features
│   ├── regimes/regime_detector.py   ← 4-regime classifier
│   ├── models/ensemble.py          ← LightGBM + Prophet + MLP + ARIMA + RF
│   ├── decisions/decision_engine.py ← Kelly-inspired sizing
│   ├── explainability/reasoning_engine.py ← 7-layer explainability
│   └── api/prophet_arena.py        ← ProphetArena adapter
├── run.py                          ← Entry point
├── requirements.txt
├── pyproject.toml
└── scripts/export.sh               ← Submission builder
```

---

## Competition Performance

- **8 market scenarios** tested: steady, trend, crash, range, chop, reversal, crisis, accumulation
- **4 regimes**: normal, volatile, trending, anomalous
- **5 models** in ensemble: LightGBM, Prophet, MLP, ARIMA, Random Forest
- **7 explainability layers** for judging transparency
- **Regime-adaptive thresholds** that adjust per market condition
- **Confidence-weighted position sizing** (Kelly-inspired, conservative variant)
- **Online regime detection** with composite z-score methodology

---

## Key Differentiators

1. **Regime Awareness**: Not just a trading bot — an adaptive forecasting intelligence
2. **Explainability**: Full reasoning chain visible at every decision
3. **Uncertainty Quantification**: 68% and 95% confidence bands on every forecast
4. **Ensemble Diversity**: 5 different model types with regime-dependent weighting
5. **Competition Ready**: ProphetArena SDK adapter included