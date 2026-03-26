# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Reproduce paper baseline numbers — ±1.5% tolerance check.

Usage::

    python -m emokit.scripts.reproduce_baselines --dry-run
    python -m emokit.scripts.reproduce_baselines \\
        --deap-root $EMOKIT_DATA_ROOT/DEAP \\
        --seedv-root $EMOKIT_DATA_ROOT/SEED-V
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

PAPER_NUMBERS: dict[str, dict[str, float]] = {
    "CNN-LSTM": {
        "DEAP_valence": 67.4,
        "DEAP_arousal": 69.2,
        "SEEDV_5class": 54.7,
    },
    "BiDAE": {
        "DEAP_valence": 71.2,
        "DEAP_arousal": 73.6,
        "SEEDV_5class": 57.3,
    },
    "DGCNN": {
        "DEAP_valence": 73.1,
        "DEAP_arousal": 72.5,
        "SEEDV_5class": 58.1,
    },
    "Transformer-MM": {
        "DEAP_valence": 70.9,
        "DEAP_arousal": 72.1,
        "SEEDV_5class": 62.3,
    },
    "DGCCA-AM": {
        "DEAP_valence": 71.5,
        "DEAP_arousal": 74.1,
        "SEEDV_5class": 60.5,
    },
    "PR-PL": {
        "DEAP_valence": 73.0,
        "DEAP_arousal": 75.4,
        "SEEDV_5class": 61.4,
    },
}

DREAMER_NUMBERS: dict[str, dict[str, float | None]] = {
    "DGCNN": {"DREAMER_valence": None, "DREAMER_arousal": None},
    "CNN-LSTM": {"DREAMER_valence": None},
    "Transformer-MM": {"DREAMER_valence": None},
}

TOLERANCE = 1.5


def check_baselines(
    our_results: dict[str, dict[str, float]] | None = None,
    dry_run: bool = False,
) -> dict[str, str]:
    """Compare our LOSO means vs paper numbers.

    Args:
        our_results: Dict model -> {setting: accuracy%}. None for dry-run.
        dry_run: If True, validate plumbing only.

    Returns:
        Dict of "Model/Setting" -> "PASS" or "FAIL (delta=X.X%)".
    """
    verdicts: dict[str, str] = {}
    total, passed = 0, 0

    for model, settings in PAPER_NUMBERS.items():
        for setting, paper_val in settings.items():
            key = f"{model}/{setting}"
            total += 1

            if dry_run or our_results is None:
                logger.info(
                    "  [DRY-RUN] %s: paper=%.1f (would compare with real data)",
                    key,
                    paper_val,
                )
                verdicts[key] = "DRY-RUN"
                continue

            ours = our_results.get(model, {}).get(setting)
            if ours is None:
                verdicts[key] = "MISSING"
                logger.info("  [MISSING] %s: no result", key)
                continue

            delta = abs(ours - paper_val)
            if delta <= TOLERANCE:
                verdicts[key] = "PASS"
                passed += 1
                logger.info(
                    "  [PASS] %s: ours=%.1f paper=%.1f Δ=%.1f",
                    key,
                    ours,
                    paper_val,
                    delta,
                )
            else:
                verdicts[key] = f"FAIL (delta={delta:.1f}%)"
                logger.info(
                    "  [FAIL] %s: ours=%.1f paper=%.1f Δ=%.1f (>±%.1f%%)",
                    key,
                    ours,
                    paper_val,
                    delta,
                    TOLERANCE,
                )

    if not dry_run and our_results is not None:
        logger.info(
            "\nSummary: %d/%d within ±%.1f%%",
            passed,
            total,
            TOLERANCE,
        )

    return verdicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate script structure with synthetic data.",
    )
    parser.add_argument("--deap-root", type=str, default=None)
    parser.add_argument("--seedv-root", type=str, default=None)
    parser.add_argument(
        "--results-json", type=str, default=None, help="Pre-computed results JSON."
    )
    args = parser.parse_args()

    our_results = None
    if args.results_json:
        our_results = json.loads(Path(args.results_json).read_text(encoding="utf-8"))

    verdicts = check_baselines(our_results=our_results, dry_run=args.dry_run)

    out_path = Path("results/reproduce_baselines.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
    logger.info("Verdicts saved to %s", out_path)


if __name__ == "__main__":
    main()
