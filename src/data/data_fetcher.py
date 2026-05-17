"""
Data Fetcher — Binance live + synthetic benchmark data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

class DataFetcher:
    """Fetches market data from Binance or generates synthetic benchmarks."""

    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol

    def fetch_live(self, interval: str = "1m", limit: int = 500) -> Optional[pd.DataFrame]:
        """Fetch live data from Binance public API (no auth needed)."""
        import urllib.request
        import json

        url = f"https://api.binance.com/api/v3/klines?symbol={self.symbol}&interval={interval}&limit={limit}"

        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read())

            df = pd.DataFrame(data, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trades", "taker_buy_base",
                "taker_buy_quote", "ignore"
            ])

            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            print(f"  [DataFetcher] Live fetch failed: {e}")
            return None

    def generate_synthetic(
        self,
        n_periods: int = 500,
        base_price: float = 50000,
        volatility: float = 0.02,
        trend: float = 0.0001,
        regime_sequence: list = None
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for benchmarking.
        Includes regime transitions to test agent adaptability.
        """
        np.random.seed(42)
        timestamps = [datetime.utcnow() - timedelta(minutes=n_periods - i) for i in range(n_periods)]

        if regime_sequence is None:
            regime_sequence = ["normal"] * 200 + ["volatile"] * 100 + ["trending"] * 150 + ["anomalous"] * 50

        prices = [base_price]
        volumes = []

        vol_map = {
            "normal": volatility,
            "volatile": volatility * 3.5,
            "trending": volatility * 1.5,
            "anomalous": volatility * 6.0
        }

        for i in range(1, n_periods):
            regime = regime_sequence[min(i, len(regime_sequence) - 1)]
            vol = vol_map[regime]

            # Geometric Brownian Motion with drift
            drift = trend if regime != "volatile" else -trend * 0.5
            shock = np.random.normal(0, 1)

            pct_change = drift + vol * shock
            new_price = prices[-1] * (1 + pct_change)
            prices.append(max(new_price, 1))

            base_vol = 1000
            vol_mult = {"normal": 1.0, "volatile": 2.5, "trending": 1.5, "anomalous": 4.0}
            volumes.append(base_vol * vol_mult[regime] * np.random.uniform(0.5, 2.0))

        # Build OHLC
        data = []
        for i, (price, vol) in enumerate(zip(prices, volumes)):
            open_p = price * (1 + np.random.uniform(-0.001, 0.001))
            close_p = price
            high_p = max(open_p, close_p) * (1 + abs(np.random.uniform(0, 0.005)))
            low_p = min(open_p, close_p) * (1 - abs(np.random.uniform(0, 0.005)))

            data.append({
                "timestamp": timestamps[i],
                "open": round(open_p, 2),
                "high": round(high_p, 2),
                "low": round(low_p, 2),
                "close": round(close_p, 2),
                "volume": round(vol, 2)
            })

        return pd.DataFrame(data)

    def fetch_for_benchmark(self, mode: str = "synthetic", **kwargs) -> pd.DataFrame:
        """
        Unified fetch interface for benchmark runner.
        mode: 'synthetic' | 'live'
        """
        if mode == "live":
            df = self.fetch_live(**kwargs)
            if df is not None:
                return df
            print("  [DataFetcher] Falling back to synthetic data...")
            mode = "synthetic"

        return self.generate_synthetic(**kwargs)