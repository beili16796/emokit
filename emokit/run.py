# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""CLI entry point: ``python -m emokit.run configs/my_config.yaml``."""

from __future__ import annotations

import argparse
import logging
from typing import Any

from emokit.evaluation.config import ConfigLoader, FullConfig
from emokit.evaluation.cross_corpus import CrossCorpusEvaluator
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
    "cross_corpus": CrossCorpusEvaluator,
}


def _dataset_kwargs(dataset_cfg: Any) -> dict[str, Any]:
    """Convert a dataset config block into ``load_dataset`` keyword arguments."""
    ds_kwargs: dict[str, Any] = {"root": dataset_cfg.root}
    if dataset_cfg.subjects is not None:
        ds_kwargs["subjects"] = dataset_cfg.subjects
    if dataset_cfg.window_sec is not None:
        ds_kwargs["window_sec"] = dataset_cfg.window_sec
    if dataset_cfg.overlap is not None:
        ds_kwargs["overlap"] = dataset_cfg.overlap
    if dataset_cfg.modalities is not None:
        ds_kwargs["modalities"] = dataset_cfg.modalities
    if dataset_cfg.label_axis is not None:
        ds_kwargs["label_axis"] = dataset_cfg.label_axis
    if hasattr(dataset_cfg, "params") and dataset_cfg.params:
        ds_kwargs.update(dataset_cfg.params)
    return ds_kwargs


def _build_evaluator(
    cfg: FullConfig,
) -> (
    LOSOEvaluator | SubjectDependentEvaluator | SessionEvaluator | CrossCorpusEvaluator
):
    """Instantiate dataset, feature pipeline, and the correct evaluator."""
    from emokit.datasets import load_dataset
    from emokit.features.base import GLOBAL_REGISTRY as TRANSFORM_REGISTRY
    from emokit.features.base import FeaturePipeline

    if cfg.model is None:
        raise ValueError("Single-model runner requires a 'model' section.")

    dataset = load_dataset(cfg.dataset.name, **_dataset_kwargs(cfg.dataset))

    steps: list[tuple[str, Any]] = []
    for step_cfg in cfg.feature_pipeline.steps:
        transform_cls = TRANSFORM_REGISTRY[step_cfg.name]
        steps.append((step_cfg.name, transform_cls(**(step_cfg.params or {}))))
    pipeline = FeaturePipeline(steps)

    protocol = cfg.evaluation.protocol
    evaluator_cls = _PROTOCOL_MAP.get(protocol)
    if evaluator_cls is None:
        raise ValueError(
            f"Unknown protocol '{protocol}'. Available: {sorted(_PROTOCOL_MAP.keys())}"
        )

    model_params = dict(cfg.model.params or {})
    if "device" not in model_params and hasattr(cfg.experiment, "device"):
        model_params["device"] = cfg.experiment.device

    if protocol == "cross_corpus":
        if cfg.target_dataset is None:
            raise ValueError(
                "cross_corpus protocol requires a top-level 'target_dataset' block."
            )
        target_dataset = load_dataset(
            cfg.target_dataset.name,
            **_dataset_kwargs(cfg.target_dataset),
        )
        return CrossCorpusEvaluator(
            source_dataset=dataset,
            target_dataset=target_dataset,
            feature_pipeline=pipeline,
            model_config=model_params,
            model_name=cfg.model.name,
            seed=cfg.experiment.seed,
            val_fraction=cfg.evaluation.val_fraction,
        )

    evaluator_kwargs = {
        "dataset": dataset,
        "feature_pipeline": pipeline,
        "model_config": model_params,
        "model_name": cfg.model.name,
        "seed": cfg.experiment.seed,
        "val_fraction": cfg.evaluation.val_fraction,
    }
    if protocol == "loso":
        evaluator_kwargs["output_config"] = dict(cfg.output.model_dump())
    return evaluator_cls(**evaluator_kwargs)


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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic data regardless of config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    args = parser.parse_args()

    logger.info("Loading config from %s", args.config)
    cfg = ConfigLoader.load(args.config)

    if args.dry_run:
        cfg = cfg.model_copy(
            update={
                "dataset": cfg.dataset.model_copy(
                    update={
                        "name": "SYNTHETIC",
                        "root": "/tmp/emokit_synthetic",
                    }
                )
            }
        )
    if args.output_dir:
        cfg = cfg.model_copy(
            update={
                "output": cfg.output.model_copy(update={"results_dir": args.output_dir})
            }
        )

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
