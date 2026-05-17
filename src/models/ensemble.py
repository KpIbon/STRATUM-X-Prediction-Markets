"""
Ensemble Forecaster — combines LightGBM + Prophet + MLP + ARIMA
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

@dataclass
class ForecastResult:
    horizon: int
    mean: float
    lower_68: float
    upper_68: float
    lower_95: float
    upper_95: float
    confidence: float
    regime: str
    model_weights: Dict[str, float]
    feature_importance: Dict[str, float]
    direction: str

@dataclass
class MarketContext:
    symbol: str = "BTC/USDT"
    current_price: float = 0.0
    timestamp: object = None
    regime: str = "normal"
    regime_confidence: float = 0.5
    volatility_zscore: float = 0.0
    trend_strength: float = 0.0
    volume_ratio: float = 1.0
    price_momentum: float = 0.0
    regime_history: List = field(default_factory=list)
    price_level: float = 0.0

class EnsembleForecaster:
    """
    Regime-adaptive ensemble forecaster.
    Combines multiple models with regime-dependent weighting.
    """

    def __init__(self):
        self.logger = logging.getLogger("Ensemble")
        self._rng = np.random.default_rng(42)

    def predict(self, features: pd.DataFrame, context: MarketContext) -> ForecastResult:
        """Generate ensemble forecast with confidence bands."""
        if len(features) < 30:
            return self._fallback_forecast(context)

        weights = self._get_regime_weights(context.regime)
        forecasts = {}

        for name, weight in weights.items():
            if weight > 0:
                fc = self._run_model(name, features, context)
                if fc is not None:
                    forecasts[name] = fc

        if not forecasts:
            return self._fallback_forecast(context)

        # Weighted mean
        total_w = sum(weights.get(n, 0) for n in forecasts)
        mean_forecast = sum(forecasts[n]["mean"] * weights.get(n, 0) for n in forecasts) / (total_w + 1e-8)

        # Variance aggregation
        variance = sum(
            weights.get(n, 0) * (forecasts[n]["std"] ** 2)
            for n in forecasts
        ) / (total_w + 1e-8)

        model_means = [forecasts[n]["mean"] for n in forecasts]
        disagreement = np.std(model_means) if len(model_means) > 1 else 0
        std = np.sqrt(variance) + disagreement * 0.5

        confidence = self._compute_confidence(forecasts, context, std)
        lower_68 = mean_forecast - 1.0 * std
        upper_68 = mean_forecast + 1.0 * std
        lower_95 = mean_forecast - 2.0 * std
        upper_95 = mean_forecast + 2.0 * std

        if mean_forecast > 0.001:
            direction = "up"
        elif mean_forecast < -0.001:
            direction = "down"
        else:
            direction = "neutral"

        importance = self._compute_importance(features)

        return ForecastResult(
            horizon=15,
            mean=mean_forecast,
            lower_68=lower_68, upper_68=upper_68,
            lower_95=lower_95, upper_95=upper_95,
            confidence=confidence,
            regime=context.regime,
            model_weights=weights,
            feature_importance=importance,
            direction=direction
        )

    def _get_regime_weights(self, regime: str) -> Dict[str, float]:
        base = {"lightgbm": 0.3, "prophet": 0.2, "mlp": 0.2, "arima": 0.15, "rf": 0.15}
        overrides = {
            "normal": {"lightgbm": 0.35, "prophet": 0.25, "mlp": 0.15, "arima": 0.15, "rf": 0.10},
            "volatile": {"lightgbm": 0.40, "prophet": 0.10, "mlp": 0.25, "arima": 0.15, "rf": 0.10},
            "trending": {"lightgbm": 0.30, "prophet": 0.30, "mlp": 0.10, "arima": 0.20, "rf": 0.10},
            "anomalous": {"lightgbm": 0.45, "mlp": 0.35, "prophet": 0.05, "arima": 0.05, "rf": 0.10}
        }
        return {**base, **overrides.get(regime, {})}

    def _run_model(self, name: str, features: pd.DataFrame, context: MarketContext) -> Optional[Dict]:
        try:
            if name == "lightgbm":
                return self._lightgbm_forecast(features, context)
            elif name == "prophet":
                return self._prophet_forecast(features, context)
            elif name == "mlp":
                return self._mlp_forecast(features, context)
            elif name == "arima":
                return self._arima_forecast(features, context)
            elif name == "rf":
                return self._rf_forecast(features, context)
        except Exception as e:
            self.logger.warning(f"Model {name} failed: {e}")
            return None

    def _lightgbm_forecast(self, features: pd.DataFrame, context: MarketContext) -> Dict:
        try:
            import lightgbm as lgb
            feature_cols = [c for c in features.columns if any(c.startswith(p) for p in ["return", "volatility", "rsi", "bb", "volume"])]
            X = features[feature_cols].values[-100:]
            y = features["close"].pct_change().dropna().values[-min(99, len(features) - 1):]
            if len(X) < 20 or len(y) < 10:
                return None
            model = lgb.LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
            model.fit(X[:-10], y[:len(X)-10])
            pred = model.predict(X[-1:])[0]
            return {"mean": pred, "std": 0.015 + abs(context.volatility_zscore) * 0.005}
        except Exception:
            return self._backup_forecast(context)

    def _prophet_forecast(self, features: pd.DataFrame, context: MarketContext) -> Dict:
        try:
            from prophet import Prophet
            close_vals = features["close"].values[-200:]
            ts = pd.DataFrame({
                "ds": pd.date_range(end=pd.Timestamp.utcnow(), periods=len(close_vals), freq="min"),
                "y": close_vals
            })
            m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=10, changepoint_prior_scale=0.05)
            m.fit(ts[-100:])
            future = m.make_future_dataframe(periods=15, freq="min")
            fc = m.predict(future)
            pred = (fc["yhat"].iloc[-1] - close_vals[-1]) / close_vals[-1]
            std = (fc["yhat_upper"].iloc[-1] - fc["yhat_lower"].iloc[-1]) / close_vals[-1]
            return {"mean": pred, "std": std}
        except Exception:
            return self._backup_forecast(context)

    def _mlp_forecast(self, features: pd.DataFrame, context: MarketContext) -> Dict:
        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
            feature_cols = [c for c in features.columns if any(c.startswith(p) for p in ["return", "volatility", "rsi", "bb"])]
            X = features[feature_cols].values[-100:]
            y = features["close"].pct_change().dropna().values[-min(99, len(features)-1):]
            if len(X) < 20:
                return None
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            model = MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu", learning_rate_init=0.01, max_iter=200, random_state=42)
            model.fit(Xs[:-10], y[:len(Xs)-10])
            pred = model.predict(Xs[-1:])[0]
            return {"mean": pred, "std": 0.02 + abs(context.volatility_zscore) * 0.01}
        except Exception:
            return self._backup_forecast(context)

    def _arima_forecast(self, features: pd.DataFrame, context: MarketContext) -> Dict:
        try:
            from statsmodels.tsa.arima.model import ARIMA
            returns = features["close"].pct_change().dropna().values[-60:]
            if len(returns) < 20:
                return self._backup_forecast(context)
            model = ARIMA(returns, order=(2, 0, 1))
            fc = model.fit()
            pred = fc.forecast(steps=15)[-1]
            return {"mean": pred, "std": np.std(returns) * 1.5}
        except Exception:
            return self._backup_forecast(context)

    def _rf_forecast(self, features: pd.DataFrame, context: MarketContext) -> Dict:
        try:
            from sklearn.ensemble import RandomForestRegressor
            feature_cols = [c for c in features.columns if any(c.startswith(p) for p in ["return", "volatility", "volume"])]
            X = features[feature_cols].values[-80:]
            y = features["close"].pct_change().dropna().values[-min(79, len(features)-1):]
            if len(X) < 20:
                return None
            model = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42, n_jobs=-1)
            model.fit(X[:-10], y[:len(X)-10])
            pred = model.predict(X[-1:])[0]
            return {"mean": pred, "std": 0.018 + abs(context.volatility_zscore) * 0.008}
        except Exception:
            return self._backup_forecast(context)

    def _backup_forecast(self, context: MarketContext) -> Dict:
        base_std = {"normal": 0.01, "volatile": 0.03, "trending": 0.015, "anomalous": 0.06}
        return {"mean": context.price_momentum * 0.001, "std": base_std.get(context.regime, 0.02)}

    def _fallback_forecast(self, context: MarketContext) -> ForecastResult:
        return ForecastResult(
            horizon=15, mean=0.0, lower_68=-0.02, upper_68=0.02,
            lower_95=-0.04, upper_95=0.04, confidence=0.3,
            regime=context.regime, model_weights={}, feature_importance={}, direction="neutral"
        )

    def _compute_confidence(self, forecasts: Dict, context: MarketContext, std: float) -> float:
        if len(forecasts) < 2:
            return 0.4
        means = [f["mean"] for f in forecasts.values()]
        disagreement = np.std(means)
        agreement = 1.0 - min(1.0, disagreement / (abs(np.mean(means)) + 0.01))
        base = 0.5 + agreement * 0.3
        bonus = {"normal": 0.1, "trending": 0.05, "volatile": -0.05, "anomalous": -0.15}
        base += bonus.get(context.regime, 0)
        if std > 0.05:
            base *= 0.8
        return max(0.1, min(0.95, base))

    def _compute_importance(self, features: pd.DataFrame) -> Dict[str, float]:
        numeric = features.select_dtypes(include=[np.number]).columns
        important = [c for c in numeric if any(c.startswith(p) for p in ["return", "volatility", "rsi", "bb", "volume"])]
        if not important:
            return {}
        return {c: round(1.0 / (i + 1), 4) for i, c in enumerate(important[:10])}