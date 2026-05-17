"""
ProphetArena Adapter — STRATUM integration with ProphetArena protocol.
Sigma Research Lab @UChicago — forecasting benchmark compatibility.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
from datetime import datetime

@dataclass
class BrierScoreResult:
    """Probabilistic calibration metric — Brier score = mean squared error of probability forecasts."""
    score: float          # 0.0 = perfect, 0.25 = random, 1.0 = worst
    calibration_error: float
    reliability: float
    resolution: float
    uncertainty: float
    n_samples: int
    is_underconfident: bool
    is_overconfident: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def grade(self) -> str:
        if self.score <= 0.05:  return "A+ Perfect"
        if self.score <= 0.10:  return "A  Well-calibrated"
        if self.score <= 0.15:  return "B  Good"
        if self.score <= 0.20:  return "C  Needs attention"
        if self.score <= 0.25:  return "D  Poor"
        return "F  Miscalibrated"


class ProphetArenaAdapter:
    """
    Converts STRATUM output to ProphetArena protocol format.
    ProphetArena is the unified benchmark for forecasting research
    at Booth School of Business under Prof. Haifeng Zhang.
    """

    @staticmethod
    def to_prophet_arena(result) -> Dict[str, Any]:
        """Convert ForecastResult → ProphetArena submission format."""
        return {
            "timestamp": result.timestamp,
            "horizon_steps": result.horizon_steps,
            "horizon_label": result.horizon_label,

            # Core probabilistic prediction
            "prediction": result.prediction,
            "prediction_lower": result.prediction_lower,
            "prediction_upper": result.prediction_upper,
            "probability": result.probability,
            "brier_score": result.brier_score,
            "calibration_score": result.calibration_score,

            # Regime classification
            "regime": result.regime.name,
            "regime_confidence": result.regime.confidence,
            "volatility": result.regime.volatility,
            "trend": result.regime.trend,
            "alpha": result.regime.alpha,
            "sigma": result.regime.sigma,

            # Decision output
            "action": result.decision.action,
            "direction": result.decision.direction,
            "decision_confidence": result.decision.confidence,
            "decision_probability": result.decision.probability,
            "decision_brier_score": result.decision.brier_score,
            "target": result.decision.target,
            "stop_loss": result.decision.stop_loss,
            "rationale": result.decision.rationale,

            # Reasoning trail
            "reasoning": [
                {"step": r.step, "heading": r.heading, "body": r.body,
                 "weight": r.weight, "type": r.type}
                for r in result.reasoning
            ],

            # Model ensemble details
            "models": [
                {
                    "name": m.model,
                    "prediction": m.prediction,
                    "probability": m.probability,
                    "brier_score": m.brier_score,
                    "is_calibrated": m.is_calibrated,
                    "feature_importance": m.feature_importance,
                }
                for m in result.models
            ],

            # System metadata
            "agent_version": result.version,
            "latency_ms": result.latency_ms,
            "protocol": "prophet-arena-v1.0",
        }

    @staticmethod
    def compute_brier_score(forecasts, actuals) -> BrierScoreResult:
        """
        Compute Brier score across multiple forecast-outcome pairs.
        Brier = mean((forecast_prob - actual_outcome)^2)
        - 0.0 = perfect probabilistic accuracy
        - 0.25 = random coin flip on binary outcomes
        - 1.0 = maximally wrong
        """
        import numpy as np

        forecasts = np.array(forecasts, dtype=float)
        actuals = np.array(actuals, dtype=float)

        if len(forecasts) != len(actuals):
            raise ValueError(f"Forecasts ({len(forecasts)}) and actuals ({len(actuals)}) must match.")

        n = len(forecasts)

        # Binary actuals (0 or 1)
        binary_outcomes = (actuals >= np.median(actuals)).astype(float)

        # Brier score
        brier = float(np.mean((forecasts - binary_outcomes) ** 2))

        # Reliability: are predicted probabilities close to observed frequencies?
        # Bin forecasts into [0-0.2], [0.2-0.4], [0.4-0.6], [0.6-0.8], [0.8-1.0]
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        reliability_list = []
        for i in range(len(bins) - 1):
            mask = (forecasts >= bins[i]) & (forecasts < bins[i+1])
            if mask.sum() > 0:
                avg_forecast = forecasts[mask].mean()
                avg_actual = binary_outcomes[mask].mean()
                reliability_list.append((avg_forecast - avg_actual) ** 2 * mask.sum())
        reliability = float(np.mean(reliability_list)) if reliability_list else 0.0

        # Resolution: how much do bin averages differ from the overall average?
        overall_avg = binary_outcomes.mean()
        resolution_list = []
        for i in range(len(bins) - 1):
            mask = (forecasts >= bins[i]) & (forecasts < bins[i+1])
            if mask.sum() > 0:
                avg_actual = binary_outcomes[mask].mean()
                resolution_list.append((avg_actual - overall_avg) ** 2 * mask.sum())
        resolution = float(np.mean(resolution_list)) if resolution_list else 0.0

        # Total uncertainty = variance of actuals
        uncertainty = float(np.var(binary_outcomes))

        # Calibration error
        calibration_error = float(np.mean(np.abs(forecasts - binary_outcomes)))

        return BrierScoreResult(
            score=round(brier, 6),
            calibration_error=round(calibration_error, 6),
            reliability=round(reliability, 6),
            resolution=round(resolution, 6),
            uncertainty=round(uncertainty, 6),
            n_samples=n,
            is_underconfident=(brier < 0.10 and calibration_error > 0.10),
            is_overconfident=(brier > 0.15 and calibration_error < 0.05),
        )

    @staticmethod
    def demo():
        print("\n" + "=" * 56)
        print("  ProphetArena Adapter — Brier Score Calibration")
        print("  Sigma Research Lab @UChicago")
        print("=" * 56)

        # Simulate 100 forecast-outcome pairs
        import numpy as np
        np.random.seed(42)
        n = 100
        forecasts = np.random.uniform(0.2, 0.8, n)
        actuals = (forecasts + np.random.randn(n) * 0.3 > 0.5).astype(float)

        brier_result = ProphetArenaAdapter.compute_brier_score(forecasts, actuals)

        print(f"\n  Brier Score:       {brier_result.score:.4f}  [{brier_result.grade()}]")
        print(f"  Calibration Err:  {brier_result.calibration_error:.4f}")
        print(f"  Reliability:      {brier_result.reliability:.4f}")
        print(f"  Resolution:       {brier_result.resolution:.4f}")
        print(f"  Uncertainty:      {brier_result.uncertainty:.4f}")
        print(f"  Samples:          {brier_result.n_samples}")
        print(f"  Underconfident:   {brier_result.is_underconfident}")
        print(f"  Overconfident:    {brier_result.is_overconfident}")

        # Quick sanity: perfect calibration
        perfect_forecasts = actuals.copy()
        perfect_result = ProphetArenaAdapter.compute_brier_score(perfect_forecasts, actuals)
        print(f"\n  Sanity check (perfect):  Brier = {perfect_result.score:.4f}")

        # Random baseline
        random_forecasts = np.full(n, 0.5)
        random_result = ProphetArenaAdapter.compute_brier_score(random_forecasts, actuals)
        print(f"  Random baseline (50%): Brier = {random_result.score:.4f}")


if __name__ == "__main__":
    ProphetArenaAdapter.demo()