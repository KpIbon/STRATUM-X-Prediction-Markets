#!/usr/bin/env python3
"""
STRATUM-X — Adaptive Forecasting Agent for Prediction Markets
Agent for ProphetArena Competition | Sigma Lab

Architecture: Regime-Aware Ensemble with Adaptive Decision Layer
Models: LightGBM + Prophet + MLP + ARIMA + XGBoost
"""

from .agents.stratum_agent import STRATUMX

__version__ = "0.1.0"
__all__ = ["STRATUMX"]