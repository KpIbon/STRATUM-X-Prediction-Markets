# STRATUM-X — ProphetArena Competition Agent
## Adaptive Forecasting Intelligence for Prediction Markets

**Institution:** Booth School of Business | University of Chicago  
**Professor:** Haifeng Zhang | Sigma Research Lab  
**Hackathon:** ProphetArena — Agents for Prediction Markets  

---

## System Overview

STRATUM-X is an adaptive forecasting intelligence system designed for real-world operational environments. It combines regime detection, ensemble forecasting, and structured reasoning to make explainable decisions in uncertain dynamic systems.

### Architecture

```
STRATUM-X
├── Frontend (Next.js 15 · React 19 · Tailwind CSS v4)
│   └── 4 pages · shadcn/ui components · Recharts visualization
├── Backend Agent (Python 3 · 8 modules)
│   ├── agents/        — Core STRATUM agent + ProphetArena SDK adapter
│   ├── data/          — Binance live data + synthetic benchmark
│   ├── features/       — 50+ engineered features (momentum, volatility, RSI, Bollinger)
│   ├── regimes/       — 4-state online classifier (normal/volatile/trending/anomalous)
│   ├── models/        — 5-model ensemble with regime-dependent weighting
│   ├── decisions/     — Kelly-inspired adaptive position sizing
│   ├── explainability/— 7-layer reasoning chain for judging panel
│   └── tests/         — Pipeline + regime detector unit tests
└── Configs (pyproject.toml, requirements.txt, run.py)
```

### Key Features

| Feature | Implementation |
|---------|---------------|
| **6 Autonomous Agents** | Data fetcher, feature engineer, regime detector, ensemble forecaster, decision engine, reasoning engine |
| **4-Regime Detection** | Online classifier: NORMAL, VOLATILE, TRENDING, ANOMALOUS |
| **5-Model Ensemble** | Linear regression, ridge, MLP, logistic regression, quantile regression — regime-weighted |
| **50+ Features** | Returns, volatility, RSI, Bollinger Bands, momentum, MACD, ATR, z-scores, EWMA |
| **7-Layer Reasoning** | Signal → Regime → Calibration → Risk → Decision → Explanation → Audit |
| **Kelly Sizing** | Fractional Kelly with regime-dependent multipliers (0.5× volatile, 1.0× trending) |
| **Brier Score** | Real-time calibration scoring on live data |
| **Confidence Bands** | 95% CI on all forecasts via quantile regression |
| **Explainability** | Full decision trail with model attribution and risk labels |

### Judging Rubric Alignment

| Criterion | Points | STRATUM-X |
|-----------|--------|-----------|
| **Creativity** | 30% | Regime-aware design, adaptive thresholds, operational framing, Kelly-inspired sizing |
| **Implementation** | 40% | Clean Python with dataclasses, 8-module architecture, ProphetArena SDK adapter |
| **Robustness** | 30% | 8 test scenarios, real-time calibration scoring, confidence bands, uncertainty-aware |

### Running the System

```bash
# Python agent
cd STRATUM-X-Prediction-Markets
pip install -r requirements.txt
python src/main.py

# Next.js frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Sample Output

```
Regime:   TRENDING [confidence=0.820, vol=1.42, trend=upward]
Forecast: 107.234 [95% CI: 107.034 — 107.434]
Probability (UP): 0.847 | Brier Score: 0.094
Decision:  LONG | up | prob=0.847 | brier=0.094 | Kelly: 0.32×
Reasoning: 5 steps | Latency: 3.2ms
  [1] Signal — Ensemble momentum bullish, 4/5 models agree
  [2] Regime — TRENDING confirmed (vol=1.42, trend=upward, confidence=0.820)
  [3] Calibration — Brier=0.094 (good), probability=0.847 well-calibrated
  [4] Risk — LOW (vol=1.42 < 2.0 threshold, regime multiplier=1.0×)
  [5] Action — LONG with Kelly fraction=0.32 (Kelly×0.4 capped)
```

### Submission Files

- `src/agents/stratum_agent.py` — Core agent
- `src/agents/prophet_arena.py` — ProphetArena SDK adapter  
- `src/data/data_fetcher.py` — Data layer
- `src/features/feature_engine.py` — Feature engineering
- `src/regimes/regime_detector.py` — Regime classifier
- `src/models/ensemble.py` — Ensemble forecaster
- `src/decisions/decision_engine.py` — Decision policy
- `src/explainability/reasoning_engine.py` — Reasoning chain
- `src/main.py` — Entry point
- `run.py` — One-command runner
- `requirements.txt` — Python dependencies
- `frontend/` — Next.js 15 application
