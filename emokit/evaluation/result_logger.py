# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Result persistence with JSON + CSV output and leaderboard support."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """JSON serialiser fallback for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class ResultLogger:
    """Persist evaluation results to JSON and CSV.

    Args:
        results_dir: Directory where results are saved.
    """

    def __init__(self, results_dir: Path | str) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save(self, results: dict, name: str) -> None:
        """Save to both JSON (full) and CSV (summary)."""
        import pandas as pd

        json_path = self.results_dir / f"{name}.json"
        csv_path = self.results_dir / f"{name}_summary.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=_json_default)

        mean = results.get("mean", {})
        std = results.get("std", {})
        rows = [
            {"metric": k, "mean": v, "std": std.get(k, "")} for k, v in mean.items()
        ]
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info("Results saved: %s, %s", json_path, csv_path)

    def append_to_leaderboard(
        self, results: dict, leaderboard_path: Path | None = None
    ) -> None:
        """Append a single row to a running leaderboard CSV."""
        import pandas as pd

        if leaderboard_path is None:
            leaderboard_path = self.results_dir / "leaderboard.csv"
        leaderboard_path = Path(leaderboard_path)

        config = results.get("config", {})
        row = {
            "timestamp": datetime.now().isoformat(),
            "model": config.get("model", config.get("model_name", "")),
            "dataset": config.get("dataset", config.get("dataset_name", "")),
            **{k: round(v, 4) for k, v in results.get("mean", {}).items()},
        }
        df = (
            pd.read_csv(leaderboard_path)
            if leaderboard_path.exists()
            else pd.DataFrame()
        )
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(leaderboard_path, index=False)
        logger.info("Updated leaderboard at %s", leaderboard_path)
