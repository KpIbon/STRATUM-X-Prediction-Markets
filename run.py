#!/usr/bin/env python3
"""
STRATUM-X — Main Entry Point
Adaptive Forecasting Agent for Prediction Markets
ProphetArena Competition | Sigma Lab

Usage:
    python run.py                    # Run benchmark
    python run.py --live             # Run with live Binance data
    python run.py --scenario STEADY_NORMAL
    python run.py --export           # Export submission package
"""

import sys
import argparse
import logging

def main():
    parser = argparse.ArgumentParser(description="STRATUM-X Forecasting Agent")
    parser.add_argument("--live", action="store_true", help="Use live Binance data")
    parser.add_argument("--scenario", type=str, help="Run specific scenario")
    parser.add_argument("--export", action="store_true", help="Export submission package")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    from src.agents.stratum_agent import STRATUMX

    print("\n" + "="*70)
    print("  STRATUM-X — Adaptive Forecasting Agent")
    print("  Prediction Markets | ProphetArena | Sigma Lab")
    print("="*70 + "\n")

    if args.scenario:
        print(f"Running scenario: {args.scenario}")
        # TODO: implement scenario-specific run
        print("Scenario runner not yet implemented — use default benchmark")

    # Run main demo/benchmark
    results = STRATUMX.demo()

    # Export if requested
    if args.export:
        from src.utils.benchmark_runner import BenchmarkRunner
        runner = BenchmarkRunner(symbol="BTC/USDT", n_rounds=10)
        report = runner.generate_report()
        print("\n" + "="*70)
        print("  BENCHMARK REPORT")
        print("="*70)
        for k, v in report.items():
            print(f"  {k}: {v}")
        print("="*70 + "\n")

    print("\n✅ STRATUM-X execution complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())