"""Config module."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DEFAULT_CONFIG = {
    "data": {
        "source": "yfinance",
        "lookback": 200,
        "symbols": ["BTC-USD", "ETH-USD"],
    },
    "features": {
        "windows": [5, 15, 30],
        "volume_window": 20,
    },
    "regime": {
        "vol_threshold": 1.5,
        "anomaly_threshold": 3.0,
        "transition_window": 10,
    },
    "forecast": {
        "n_estimators": 100,
        "max_depth": 6,
    },
    "decision": {
        "stable": {"buy_thresh": 0.55, "sell_thresh": 0.45, "conf_min": 0.65},
        "volatile": {"buy_thresh": 0.65, "sell_thresh": 0.35, "conf_min": 0.70},
        "transition": {"buy_thresh": 0.60, "sell_thresh": 0.40, "conf_min": 0.75},
        "anomalous": {"buy_thresh": 0.75, "sell_thresh": 0.25, "conf_min": 0.80},
    },
}