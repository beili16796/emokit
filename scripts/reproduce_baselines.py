# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Compare LOSO (or held-out) metrics against published numbers — fill in after real runs.

Usage (after you have real experiment outputs)::

    python scripts/reproduce_baselines.py --ours results/loso_deap_dgcnn.json

This script is a **template**: wire ``run_loso`` to your saved JSON from
``LOSOEvaluator`` or your logging pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Placeholder literature / report targets — replace with your paper's citations.
PAPER_NUMBERS: dict[str, dict[str, float]] = {
    "DGCNN": {"deap_valence": 73.1, "deap_arousal": 72.5},
    "PR-PL": {"deap_valence": 73.0, "deap_arousal": 75.4},
}

TOLERANCE_PCT = 1.5


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Entry point — compare ours vs paper placeholders."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ours",
        type=str,
        help="Path to JSON with keys like mean.accuracy per model (optional).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DGCNN",
        help="Model name key in PAPER_NUMBERS.",
    )
    args = parser.parse_args()

    paper = PAPER_NUMBERS.get(args.model)
    if paper is None:
        logger.error("Unknown model %r — add to PAPER_NUMBERS.", args.model)
        sys.exit(1)

    if not args.ours:
        logger.info(
            "No --ours file provided. This script is a template.\n"
            "Expected JSON shape (example): "
            '{"mean": {"accuracy": 72.0}, "std": {"accuracy": 2.1}} '
            "or your own schema — adjust mapping below.",
        )
        for metric, paper_val in paper.items():
            logger.info("Paper %s %s: %.1f (target ±%.1f%%)", args.model, metric, paper_val, TOLERANCE_PCT)
        sys.exit(0)

    ours_path = Path(args.ours)
    data = _load_json(ours_path)
    # TODO: map your JSON to metric names (deap_valence / deap_arousal).
    ours_map = {
        "deap_valence": float(data.get("mean", {}).get("accuracy", 0.0)),
        "deap_arousal": float(data.get("mean", {}).get("accuracy", 0.0)),
    }

    for metric, paper_val in paper.items():
        our_val = ours_map.get(metric)
        if our_val is None:
            continue
        delta = abs(our_val - paper_val)
        status = "PASS" if delta <= TOLERANCE_PCT else "FAIL"
        logger.info(
            "%s %s: ours=%.2f paper=%.1f Δ=%.2f [%s]",
            args.model,
            metric,
            our_val,
            paper_val,
            delta,
            status,
        )


if __name__ == "__main__":
    main()
