#!/usr/bin/env python3
"""Augmentation ablation: compare DGCNN on DEAP with/without augmentations.

Conditions:
  1. No augmentation (baseline)
  2. FeatureMixup only
  3. TemporalSegmentPermutation only
  4. Both

Usage::

    python -m emokit.scripts.augmentation_ablation \
        --root $EMOKIT_DATA_ROOT/DEAP \
        --device cuda \
        --output results/augmentation_ablation.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONDITIONS = [
    ("No augmentation", []),
    ("FeatureMixup", [("FeatureMixup", {"alpha": 0.4, "p": 0.5, "seed": 42})]),
    (
        "TemporalPerm",
        [("TemporalSegmentPermutation", {"n_segments": 4, "p": 0.5, "seed": 42})],
    ),
    (
        "Both",
        [
            ("FeatureMixup", {"alpha": 0.4, "p": 0.5, "seed": 42}),
            ("TemporalSegmentPermutation", {"n_segments": 4, "p": 0.5, "seed": 42}),
        ],
    ),
]


def run_ablation(
    root: str,
    device: str = "cpu",
    output: str = "results/augmentation_ablation.json",
    label_axis: str = "valence",
    selected_conditions: list[str] | None = None,
) -> dict[str, Any]:
    from emokit.datasets import load_dataset
    from emokit.evaluation.protocols import LOSOEvaluator
    from emokit.features.augmentation import FeatureMixup, TemporalSegmentPermutation
    from emokit.features.base import FeaturePipeline
    from emokit.features.eeg import DEExtractor, EEGNormalizer
    from emokit.utils import set_seed

    set_seed(42)

    use_synthetic = root is None or not Path(root).exists()
    if use_synthetic:
        logger.warning("Data root not found (%s), using SYNTHETIC.", root)

    results: dict[str, Any] = {}
    output_root = Path(output).with_suffix("")

    condition_lookup = {name: aug_specs for name, aug_specs in CONDITIONS}
    if selected_conditions:
        unknown = sorted(set(selected_conditions) - set(condition_lookup))
        if unknown:
            raise ValueError(f"Unknown augmentation conditions: {unknown}")
        conditions = [(name, condition_lookup[name]) for name in selected_conditions]
    else:
        conditions = CONDITIONS

    for cond_name, aug_specs in conditions:
        logger.info("=== Condition: %s ===", cond_name)

        steps: list[tuple[str, Any]] = [
            ("de", DEExtractor(fs=128)),
            ("norm", EEGNormalizer()),
        ]
        for aug_name, aug_params in aug_specs:
            if aug_name == "FeatureMixup":
                steps.append(("mixup", FeatureMixup(**aug_params)))
            elif aug_name == "TemporalSegmentPermutation":
                steps.append(("tempperm", TemporalSegmentPermutation(**aug_params)))

        pipeline = FeaturePipeline(steps)

        if use_synthetic:
            ds = load_dataset("SYNTHETIC")
        else:
            ds = load_dataset(
                "DEAP",
                root=root,
                label_axis=label_axis,
                label_threshold=5.0,
            )

        evaluator = LOSOEvaluator(
            dataset=ds,
            feature_pipeline=pipeline,
            model_config={
                "n_classes": 2,
                "n_channels": 32,
                "n_bands": 5,
                "K": 2,
                "n_epochs": 50,
                "device": device,
            },
            model_name="DGCNN",
            seed=42,
            output_config={
                "results_dir": str(output_root / cond_name.replace(" ", "_")),
                "save_checkpoints": False,
            },
        )

        try:
            result = evaluator.run()
            mean_acc = result["mean"]["accuracy"] * 100
            std_acc = result["std"]["accuracy"] * 100
            results[cond_name] = {
                "mean_acc": round(mean_acc, 2),
                "std_acc": round(std_acc, 2),
                "per_subject": {
                    str(k): round(v["accuracy"] * 100, 2)
                    for k, v in result.get("per_subject", {}).items()
                },
            }
            logger.info("%s: %.1f ± %.1f%%", cond_name, mean_acc, std_acc)
        except Exception as e:
            logger.exception("Failed %s: %s", cond_name, e)
            results[cond_name] = {"mean_acc": 0.0, "std_acc": 0.0, "error": str(e)}

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", out_path)

    print("\n" + "=" * 55)
    print(f"AUGMENTATION ABLATION — DGCNN / DEAP / {label_axis.title()} (LOSO)")
    print("=" * 55)
    for cond_name in [c[0] for c in conditions]:
        r = results.get(cond_name, {})
        print(
            f"  {cond_name:25s}: "
            f"{r.get('mean_acc', 0):.1f} ± {r.get('std_acc', 0):.1f}%"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output",
        type=str,
        default="results/augmentation_ablation.json",
    )
    parser.add_argument(
        "--label-axis",
        type=str,
        default="valence",
        choices=["valence", "arousal"],
    )
    parser.add_argument(
        "--conditions",
        nargs="*",
        default=None,
        choices=[name for name, _ in CONDITIONS],
        help="Subset of augmentation conditions to run.",
    )
    args = parser.parse_args()
    run_ablation(
        root=args.root,
        device=args.device,
        output=args.output,
        label_axis=args.label_axis,
        selected_conditions=args.conditions,
    )


if __name__ == "__main__":
    main()
