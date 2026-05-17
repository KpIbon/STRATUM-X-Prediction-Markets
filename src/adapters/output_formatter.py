"""
Output Formatter — saves agent results to JSON/CSV.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List

log = logging.getLogger("stratum.adapter.output")


class OutputFormatter:
    """Saves results in ProphetArena-compatible formats."""

    def save(self, result: dict, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".json":
            with open(p, "w") as f:
                json.dump(result, f, indent=2, default=str)
        else:
            with open(p, "w") as f:
                json.dump(result, f, indent=2, default=str)
        log.info(f"Saved output to {p}")

    def save_batch(self, results: List[dict], path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"Saved {len(results)} batch results to {p}")

    def to_csv(self, result: dict, path: str) -> None:
        """Export a summary CSV for quick review."""
        row = {
            "symbol": result.get("metadata", {}).get("symbol", ""),
            "forecast": result.get("forecast", {}).get("value", ""),
            "confidence": result.get("forecast", {}).get("confidence", ""),
            "regime": result.get("regime", {}).get("state", ""),
            "action": result.get("decision", {}).get("action", ""),
            "timestamp": result.get("metadata", {}).get("timestamp", ""),
            "status": result.get("metadata", {}).get("status", ""),
        }
        file_exists = Path(path).exists()
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)