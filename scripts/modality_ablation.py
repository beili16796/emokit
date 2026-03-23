# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Modality ablation experiment template.

Systematically removes one modality at a time and re-runs LOSO evaluation
to quantify each modality's contribution.

Usage::

    python scripts/modality_ablation.py --config configs/deap_loso_dgcnn.yaml

Override the full modality list or the model from the command line::

    python scripts/modality_ablation.py --config configs/deap_loso_dgcnn.yaml \
        --modalities eeg gsr ecg --model DGCNN
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_ablation(
    base_config_path: str,
    modalities: list[str] | None = None,
    model_name: str | None = None,
    results_dir: str = "results/ablation/",
) -> dict[str, Any]:
    """Run leave-one-modality-out ablation.

    For *N* modalities the experiment runs *N + 1* LOSO evaluations:
    the full set plus *N* subsets each missing one modality.

    Args:
        base_config_path: Path to a YAML experiment config.
        modalities: Explicit modality list; inferred from config if ``None``.
        model_name: Override model name.
        results_dir: Where to write per-condition JSON results.

    Returns:
        Summary dict mapping condition names to mean accuracy / F1.
    """
    from emokit.datasets import load_dataset
    from emokit.evaluation.protocols import LOSOEvaluator
    from emokit.features.base import GLOBAL_REGISTRY as TRANSFORM_REGISTRY, FeaturePipeline
    from emokit.utils import set_seed

    cfg = _load_yaml(base_config_path)
    seed = cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    ds_cfg = cfg["dataset"]
    if modalities is None:
        modalities = ds_cfg.get("modalities") or ["eeg"]
    if model_name is None:
        model_name = cfg["model"]["name"]
    model_params = cfg["model"].get("params", {})

    steps_cfg = cfg.get("feature_pipeline", {}).get("steps", [])
    steps: list[tuple[str, Any]] = []
    for s in steps_cfg:
        cls = TRANSFORM_REGISTRY[s["name"]]
        steps.append((s["name"], cls(**(s.get("params") or {}))))

    val_fraction = cfg.get("evaluation", {}).get("val_fraction", 0.1)
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions: list[tuple[str, list[str]]] = [
        ("all_modalities", list(modalities)),
    ]
    for drop in modalities:
        subset = [m for m in modalities if m != drop]
        if subset:
            conditions.append((f"drop_{drop}", subset))

    summary: dict[str, dict[str, float]] = {}

    for condition_name, mod_subset in conditions:
        logger.info("=== Condition: %s  modalities=%s ===", condition_name, mod_subset)

        ds_kwargs: dict[str, Any] = {"root": ds_cfg.get("root", "data/")}
        if ds_cfg.get("subjects"):
            ds_kwargs["subjects"] = ds_cfg["subjects"]
        if ds_cfg.get("window_sec"):
            ds_kwargs["window_sec"] = ds_cfg["window_sec"]
        if ds_cfg.get("overlap") is not None:
            ds_kwargs["overlap"] = ds_cfg["overlap"]
        ds_kwargs["modalities"] = mod_subset
        if ds_cfg.get("label_axis"):
            ds_kwargs["label_axis"] = ds_cfg["label_axis"]

        dataset = load_dataset(ds_cfg["name"], **ds_kwargs)

        pipeline = FeaturePipeline(
            [(n, type(t)(**{}) if hasattr(t, "__dict__") else t) for n, t in steps]
        )

        evaluator = LOSOEvaluator(
            dataset=dataset,
            feature_pipeline=pipeline,
            model_config=model_params,
            model_name=model_name,
            seed=seed,
            val_fraction=val_fraction,
        )
        results = evaluator.run()

        json_path = out_dir / f"{condition_name}.json"
        json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        logger.info("Saved %s", json_path)

        summary[condition_name] = {
            "mean_accuracy": results.get("mean", {}).get("accuracy", 0.0),
            "mean_f1_macro": results.get("mean", {}).get("f1_macro", 0.0),
        }

    logger.info("\n%s", "=" * 60)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 60)
    for cond, metrics in summary.items():
        logger.info(
            "  %-25s  acc=%.4f  f1=%.4f",
            cond,
            metrics["mean_accuracy"],
            metrics["mean_f1_macro"],
        )

    summary_path = out_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary written to %s", summary_path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Base YAML experiment config.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help="Override modality list (space-separated).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/ablation/",
        help="Output directory for ablation results.",
    )
    args = parser.parse_args()

    run_ablation(
        base_config_path=args.config,
        modalities=args.modalities,
        model_name=args.model,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
