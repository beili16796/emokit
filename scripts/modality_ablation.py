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
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run leave-one-modality-out ablation.

    For *N* modalities the experiment runs *N + 1* LOSO evaluations:
    the full set plus *N* subsets each missing one modality.

    Args:
        base_config_path: Path to a YAML experiment config.
        modalities: Explicit modality list; inferred from config if ``None``.
        model_name: Override model name.
        results_dir: Where to write per-condition JSON results.
        dry_run: If ``True``, substitute SyntheticDataset for real data.

    Returns:
        Summary dict mapping condition names to mean accuracy / F1.
    """

    from emokit.datasets import load_dataset
    from emokit.datasets.synthetic import SyntheticDataset
    from emokit.evaluation.protocols import LOSOEvaluator
    from emokit.features.base import (
        GLOBAL_REGISTRY as TRANSFORM_REGISTRY,
    )
    from emokit.features.base import (
        BaseTransform,
        FeaturePipeline,
    )
    from emokit.features.eeg import DEExtractor, EEGNormalizer
    from emokit.utils import set_seed

    cfg = _load_yaml(base_config_path)
    seed = cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    ds_cfg = cfg["dataset"]
    if modalities is None:
        modalities = ds_cfg.get("modalities") or ["eeg"]
    if model_name is None:
        model_name = cfg["model"]["name"]
    model_params = dict(cfg["model"].get("params", {}))

    if dry_run:
        model_params["n_epochs"] = 2

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

        if dry_run:
            n_channels = model_params.get("n_channels", 32)
            dataset = SyntheticDataset(
                n_subjects=3,
                n_trials=8,
                n_channels=n_channels,
                n_classes=2,
                modalities=mod_subset,
                fs=128.0,
                window_sec=4.0,
            )
        else:
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

        if dry_run:

            class _ReshapeTo3D(BaseTransform):
                def __init__(self, nc, ns):
                    self.nc, self.ns = nc, ns
                def fit(self, X, y=None):
                    return self
                def transform(self, X):
                    X = np.asarray(X)
                    return X.reshape(X.shape[0], self.nc, self.ns) if X.ndim == 2 else X

            class _FlattenTo2D(BaseTransform):
                def fit(self, X, y=None):
                    return self
                def transform(self, X):
                    X = np.asarray(X)
                    return X.reshape(X.shape[0], -1) if X.ndim == 3 else X

            n_ch_pipe = model_params.get("n_channels", 32)
            n_samples_pipe = int(128.0 * 4.0)
            needs_flat = model_name not in ("DGCNN",)
            dry_steps = [
                ("reshape", _ReshapeTo3D(n_ch_pipe, n_samples_pipe)),
                ("de", DEExtractor(fs=128)),
                ("norm", EEGNormalizer()),
            ]
            if needs_flat:
                dry_steps.append(("flatten", _FlattenTo2D()))
            pipeline = FeaturePipeline(dry_steps)
        else:
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
        json_path.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use SyntheticDataset instead of real data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the ablation summary JSON (overrides --results-dir).",
    )
    args = parser.parse_args()

    summary = run_ablation(
        base_config_path=args.config,
        modalities=args.modalities,
        model_name=args.model,
        results_dir=args.results_dir,
        dry_run=args.dry_run,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Output also written to %s", out_path)


if __name__ == "__main__":
    main()
