"""Config module."""

from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_estimators: int = 50
    max_depth: int = 6
    seed: int = 42
    calibration_samples: int = 100