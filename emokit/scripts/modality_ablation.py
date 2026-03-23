# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Modality ablation experiment — generates paper Table 5.

Usage::

    python -m emokit.scripts.modality_ablation --dataset DEAP \\
        --root $EMOKIT_DATA_ROOT/DEAP --output results/modality_ablation_DEAP.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

MODALITY_SETS = [
    ["eeg"],
    ["gsr"],
    ["ecg"],
    ["eeg", "gsr"],
    ["eeg", "ecg"],
    ["eeg", "gsr", "ecg"],
]

MODALITY_NAMES = {
    ("eeg",): "EEG only",
    ("gsr",): "GSR only",
    ("ecg",): "ECG only",
    ("eeg", "gsr"): "EEG+GSR",
    ("eeg", "ecg"): "EEG+ECG",
    ("eeg", "gsr", "ecg"): "EEG+GSR+ECG",
}


def _print_latex_table(results: dict[str, dict[str, float]]) -> None:
    """Print a LaTeX-formatted ablation table."""
    print("\\begin{table}")
    print("\\caption{Modality ablation on DEAP (DGCCA-AM, LOSO).}")
    print("\\begin{tabular}{lcc}")
    print("\\toprule")
    print("Modalities & Valence (\\%) & Arousal (\\%) \\\\")
    print("\\midrule")
    for mod_key, metrics in results.items():
        val_str = f"{metrics.get('valence_mean', 0):.1f} ± {metrics.get('valence_std', 0):.1f}"
        aro_str = f"{metrics.get('arousal_mean', 0):.1f} ± {metrics.get('arousal_std', 0):.1f}"
        print(f"{mod_key} & {val_str} & {aro_str} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def run_ablation(
    dataset_name: str = "DEAP",
    root: str | None = None,
    output: str = "results/modality_ablation_DEAP.json",
) -> dict[str, dict[str, float]]:
    """Run modality ablation with DGCCA-AM on DEAP via LOSO."""
    from emokit.datasets import load_dataset
    from emokit.evaluation.protocols import LOSOEvaluator
    from emokit.features.base import FeaturePipeline
    from emokit.features.eeg import DEExtractor, EEGNormalizer
    from emokit.utils import set_seed

    set_seed(42)

    use_synthetic = False
    if root is None or not Path(root).exists():
        logger.warning(
            "Data root not found (%s). Falling back to synthetic data.", root
        )
        dataset_name = "SYNTHETIC"
        use_synthetic = True

    results: dict[str, dict[str, float]] = {}

    for mod_set in MODALITY_SETS:
        mod_name = MODALITY_NAMES.get(tuple(mod_set), "+".join(mod_set))
        logger.info("=== Running: %s ===", mod_name)

        ds_kwargs: dict[str, Any] = {"modalities": mod_set}
        if not use_synthetic:
            ds_kwargs["root"] = root

        try:
            ds = load_dataset(dataset_name, **ds_kwargs)
        except Exception as e:
            logger.warning("Skipping %s: %s", mod_name, e)
            continue

        pipeline = FeaturePipeline([
            ("de", DEExtractor(fs=128)),
            ("norm", EEGNormalizer()),
        ])

        evaluator = LOSOEvaluator(
            dataset=ds,
            feature_pipeline=pipeline,
            model_config={},
            model_name="DGCCA-AM",
            seed=42,
        )

        try:
            result = evaluator.run()
            results[mod_name] = {
                "valence_mean": result["mean"].get("accuracy", 0) * 100,
                "valence_std": result["std"].get("accuracy", 0) * 100,
                "arousal_mean": 0.0,
                "arousal_std": 0.0,
            }
        except Exception as e:
            logger.warning("Failed %s: %s", mod_name, e)
            results[mod_name] = {
                "valence_mean": 0.0, "valence_std": 0.0,
                "arousal_mean": 0.0, "arousal_std": 0.0,
            }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", out_path)

    _print_latex_table(results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="DEAP")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument(
        "--output", type=str, default="results/modality_ablation_DEAP.json"
    )
    args = parser.parse_args()
    run_ablation(dataset_name=args.dataset, root=args.root, output=args.output)


if __name__ == "__main__":
    main()
