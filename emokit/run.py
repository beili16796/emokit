# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""CLI entry point: ``python -m emokit.run configs/my_config.yaml``."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

from emokit.evaluation.config import ConfigLoader, FullConfig
from emokit.evaluation.protocols import (
    LOSOEvaluator,
    ResultLogger,
    SessionEvaluator,
    SubjectDependentEvaluator,
)
from emokit.utils import set_seed

logger = logging.getLogger(__name__)

_PROTOCOL_MAP: dict[str, type] = {
    "loso": LOSOEvaluator,
    "subject_dependent": SubjectDependentEvaluator,
    "session": SessionEvaluator,
}


def _build_evaluator(cfg: FullConfig) -> LOSOEvaluator | SubjectDependentEvaluator | SessionEvaluator:
    """Instantiate dataset, feature pipeline, and the correct evaluator."""
    from emokit.datasets import load_dataset
    from emokit.features.base import GLOBAL_REGISTRY as TRANSFORM_REGISTRY
    from emokit.features.base import FeaturePipeline

    ds_kwargs: dict[str, Any] = {"root": cfg.dataset.root}
    if cfg.dataset.subjects is not None:
        ds_kwargs["subjects"] = cfg.dataset.subjects
    if cfg.dataset.window_sec is not None:
        ds_kwargs["window_sec"] = cfg.dataset.window_sec
    if cfg.dataset.overlap is not None:
        ds_kwargs["overlap"] = cfg.dataset.overlap
    if cfg.dataset.modalities is not None:
        ds_kwargs["modalities"] = cfg.dataset.modalities
    if cfg.dataset.label_axis is not None:
        ds_kwargs["label_axis"] = cfg.dataset.label_axis

    dataset = load_dataset(cfg.dataset.name, **ds_kwargs)

    steps: list[tuple[str, Any]] = []
    for step_cfg in cfg.feature_pipeline.steps:
        transform_cls = TRANSFORM_REGISTRY[step_cfg.name]
        steps.append((step_cfg.name, transform_cls(**(step_cfg.params or {}))))
    pipeline = FeaturePipeline(steps)

    protocol = cfg.evaluation.protocol
    evaluator_cls = _PROTOCOL_MAP.get(protocol)
    if evaluator_cls is None:
        raise ValueError(
            f"Unknown protocol '{protocol}'. "
            f"Available: {sorted(_PROTOCOL_MAP.keys())}"
        )

    return evaluator_cls(
        dataset=dataset,
        feature_pipeline=pipeline,
        model_config=cfg.model.params or {},
        model_name=cfg.model.name,
        seed=cfg.experiment.seed,
        val_fraction=cfg.evaluation.val_fraction,
    )


def _print_summary(results: dict[str, Any]) -> None:
    """Print a human-readable summary table to stdout."""
    config = results.get("config", {})
    mean = results.get("mean", {})
    std = results.get("std", {})

    header = (
        f"\n{'=' * 60}\n"
        f"  Experiment: {config.get('dataset_name', '?')} / "
        f"{config.get('model_name', '?')} / "
        f"{config.get('protocol', '?')}\n"
        f"  Seed: {config.get('seed', '?')}\n"
        f"{'=' * 60}"
    )
    logger.info(header)
    print(header)

    row_fmt = "  {:<20s}  {:.4f} ± {:.4f}"
    for key in sorted(mean.keys()):
        line = row_fmt.format(key, mean[key], std.get(key, 0.0))
        logger.info(line)
        print(line)

    n_subjects = len(results.get("per_subject", {}))
    footer = f"\n  Evaluated {n_subjects} subject(s).\n{'=' * 60}\n"
    logger.info(footer)
    print(footer)


def main() -> None:
    """Parse CLI arguments and run the experiment pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="EmoKit experiment runner",
        prog="python -m emokit.run",
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to the experiment YAML configuration file.",
    )
    args = parser.parse_args()

    logger.info("Loading config from %s", args.config)
    cfg = ConfigLoader.load(args.config)

    set_seed(cfg.experiment.seed)
    logger.info("Experiment: %s  seed=%d", cfg.experiment.name, cfg.experiment.seed)

    evaluator = _build_evaluator(cfg)
    results = evaluator.run()

    result_logger = ResultLogger(results_dir=cfg.output.results_dir)
    json_path = result_logger.log(results)
    logger.info("Results written to %s", json_path)

    _print_summary(results)


if __name__ == "__main__":
    main()
