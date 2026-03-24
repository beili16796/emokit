# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Master script: run every experiment YAML and collect all results.

Usage::

    python -m emokit.scripts.run_all_experiments \\
        --output-dir results/full_run \\
        --deap-root data/DEAP --seed-root data/SEED ...

Dry-run mode substitutes SyntheticDataset for every real dataset::

    python -m emokit.scripts.run_all_experiments --dry-run \\
        --output-dir /tmp/emokit_dryrun
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

EXPERIMENT_CONFIGS: list[dict[str, Any]] = [
    {
        "name": "deap_loso_dgcnn_valence",
        "dataset": "DEAP",
        "model": "DGCNN",
        "label_axis": "valence",
        "modalities": ["eeg"],
        "model_params": {
            "n_channels": 32,
            "n_bands": 5,
            "hidden_dim": 64,
            "n_epochs": 50,
            "lr": 0.001,
            "batch_size": 64,
            "lambda_reg": 0.0001,
        },
    },
    {
        "name": "deap_loso_dgcnn_arousal",
        "dataset": "DEAP",
        "model": "DGCNN",
        "label_axis": "arousal",
        "modalities": ["eeg"],
        "model_params": {
            "n_channels": 32,
            "n_bands": 5,
            "hidden_dim": 64,
            "n_epochs": 50,
            "lr": 0.001,
            "batch_size": 64,
            "lambda_reg": 0.0001,
        },
    },
    {
        "name": "deap_loso_cnnlstm_valence",
        "dataset": "DEAP",
        "model": "CNN-LSTM",
        "label_axis": "valence",
        "modalities": ["eeg"],
        "model_params": {
            "n_classes": 2,
            "input_type": "de",
            "n_channels": 32,
            "n_epochs": 50,
            "batch_size": 64,
            "hidden_size": 64,
            "n_layers": 2,
            "dropout": 0.3,
            "lr": 1e-3,
            "device": "cpu",
        },
    },
    {
        "name": "deap_loso_prpl_valence",
        "dataset": "DEAP",
        "model": "PR-PL",
        "label_axis": "valence",
        "modalities": ["eeg"],
        "model_params": {
            "n_classes": 2,
            "n_feat": 160,
            "n_epochs": 50,
            "batch_size": 64,
            "lr": 1e-3,
        },
    },
    {
        "name": "seedv_loso_dgcnn",
        "dataset": "SEED-V",
        "model": "DGCNN",
        "label_axis": None,
        "modalities": ["eeg"],
        "model_params": {
            "n_channels": 62,
            "n_bands": 5,
            "hidden_dim": 64,
            "n_epochs": 50,
            "lr": 0.001,
            "batch_size": 64,
            "lambda_reg": 0.0001,
        },
    },
    {
        "name": "seedv_loso_transformer",
        "dataset": "SEED-V",
        "model": "Transformer-MM",
        "label_axis": None,
        "modalities": ["eeg"],
        "model_params": {
            "n_channels": 62,
            "n_bands": 5,
            "d_model": 128,
            "nhead": 8,
            "n_layers": 4,
            "dropout": 0.1,
            "lr": 0.0005,
            "batch_size": 64,
            "n_epochs": 80,
        },
    },
    {
        "name": "seed_loso_dgcnn",
        "dataset": "SEED",
        "model": "DGCNN",
        "label_axis": None,
        "modalities": ["eeg"],
        "model_params": {
            "n_channels": 62,
            "n_bands": 5,
            "hidden_dim": 64,
            "n_epochs": 50,
            "lr": 0.001,
            "batch_size": 64,
            "lambda_reg": 0.0001,
        },
    },
    {
        "name": "dreamer_loso_dgcnn_valence",
        "dataset": "DREAMER",
        "model": "DGCNN",
        "label_axis": "valence",
        "modalities": ["eeg"],
        "model_params": {
            "n_channels": 14,
            "n_bands": 5,
            "hidden_dim": 64,
            "n_epochs": 50,
            "lr": 0.001,
            "batch_size": 64,
            "lambda_reg": 0.0001,
        },
    },
    {
        "name": "hci_loso_dgcnn_valence",
        "dataset": "MAHNOB-HCI",
        "model": "DGCNN",
        "label_axis": "valence",
        "modalities": ["eeg"],
        "model_params": {
            "n_channels": 32,
            "n_bands": 5,
            "hidden_dim": 64,
            "n_epochs": 50,
            "lr": 0.001,
            "batch_size": 64,
            "lambda_reg": 0.0001,
        },
    },
]


class _ReshapeTo3D:
    """Reshape ``(N, C*T)`` back to ``(N, C, T)`` for transforms needing 3D."""

    def __init__(self, n_channels: int, n_samples: int) -> None:
        self.n_channels = n_channels
        self.n_samples = n_samples

    def fit(self, X: Any, y: Any = None) -> _ReshapeTo3D:
        return self

    def transform(self, X: Any) -> Any:
        X = np.asarray(X)
        if X.ndim == 2:
            return X.reshape(X.shape[0], self.n_channels, self.n_samples)
        return X

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        return self.fit(X, y).transform(X)


class _FlattenTo2D:
    """Flatten ``(N, C, F)`` to ``(N, C*F)``."""

    def fit(self, X: Any, y: Any = None) -> _FlattenTo2D:
        return self

    def transform(self, X: Any) -> Any:
        X = np.asarray(X)
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X

    def fit_transform(self, X: Any, y: Any = None) -> Any:
        return self.fit(X, y).transform(X)


def _make_synthetic_dataset(
    n_subjects: int = 3,
    n_trials: int = 10,
    n_channels: int = 32,
    n_classes: int = 2,
    modalities: list[str] | None = None,
    fs: float = 128.0,
    window_sec: float = 4.0,
) -> Any:
    """Create a SyntheticDataset for dry-run mode."""
    from emokit.datasets.synthetic import SyntheticDataset

    return SyntheticDataset(
        n_subjects=n_subjects,
        n_trials=n_trials,
        n_channels=n_channels,
        n_classes=n_classes,
        modalities=modalities,
        fs=fs,
        window_sec=window_sec,
    )


def _run_single(
    exp: dict[str, Any],
    dry_run: bool,
    output_dir: Path,
    data_roots: dict[str, str],
) -> dict[str, Any]:
    """Run a single experiment configuration."""
    from emokit.datasets import load_dataset
    from emokit.evaluation.protocols import LOSOEvaluator
    from emokit.features.base import FeaturePipeline
    from emokit.features.eeg import DEExtractor, EEGNormalizer
    from emokit.utils import set_seed

    set_seed(42)
    name = exp["name"]
    model_name = exp["model"]
    model_params = dict(exp["model_params"])
    modalities = exp.get("modalities", ["eeg"])

    fs = 128.0
    window_sec = 4.0

    if dry_run:
        n_ch = model_params.get("n_channels", 32)

        if model_name == "Transformer-MM":
            n_bands = model_params.get("n_bands", 5)
            model_params["n_peripheral_feat"] = n_ch
            ds = _make_synthetic_dataset(
                n_subjects=3,
                n_trials=8,
                n_channels=n_ch,
                n_classes=2,
                modalities=modalities,
                fs=float(n_bands),
                window_sec=1.0,
            )
        else:
            ds = _make_synthetic_dataset(
                n_subjects=3,
                n_trials=8,
                n_channels=n_ch,
                n_classes=2,
                modalities=modalities,
                fs=fs,
                window_sec=window_sec,
            )

        model_params["n_epochs"] = 2
        if "n_feat" in model_params:
            model_params["n_feat"] = n_ch * 5
    else:
        root_key = exp["dataset"].replace("-", "_").upper()
        root = data_roots.get(root_key)
        ds_kwargs: dict[str, Any] = {"modalities": modalities}
        if root:
            ds_kwargs["root"] = root
        if exp.get("label_axis"):
            ds_kwargs["label_axis"] = exp["label_axis"]
        ds = load_dataset(exp["dataset"], **ds_kwargs)

    if dry_run:
        n_samples = int(fs * window_sec)
        n_ch = model_params.get("n_channels", 32)

        needs_flat_output = model_name in ("CNN-LSTM", "PR-PL")
        is_multimodal_model = model_name in ("Transformer-MM", "BiDAE", "DGCCA-AM")

        if is_multimodal_model:
            from emokit.features.base import BaseTransform as _BT

            class _Identity(_BT):
                def fit(self, X, y=None):
                    return self
                def transform(self, X):
                    return X

            pipeline = FeaturePipeline([("identity", _Identity())])
        else:
            steps: list[tuple[str, Any]] = [
                ("reshape", _ReshapeTo3D(n_ch, n_samples)),
                ("de", DEExtractor(fs=int(fs))),
                ("norm", EEGNormalizer()),
            ]
            if needs_flat_output:
                steps.append(("flatten", _FlattenTo2D()))
            pipeline = FeaturePipeline(steps)
    else:
        pipeline = FeaturePipeline([
            ("de", DEExtractor(fs=int(fs))),
            ("norm", EEGNormalizer()),
        ])

    evaluator = LOSOEvaluator(
        dataset=ds,
        feature_pipeline=pipeline,
        model_config=model_params,
        model_name=model_name,
        seed=42,
        val_fraction=0.1,
    )

    logger.info("Running experiment: %s", name)
    results = evaluator.run()

    json_path = output_dir / f"{name}.json"
    json_path.write_text(
        json.dumps(results, indent=2, default=_json_default),
        encoding="utf-8",
    )
    logger.info("Saved %s", json_path)

    return {
        "name": name,
        "mean_accuracy": results.get("mean", {}).get("accuracy", 0.0),
        "mean_f1_macro": results.get("mean", {}).get("f1_macro", 0.0),
    }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use SyntheticDataset for all experiments.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/full_run/",
        help="Directory for per-experiment JSON results.",
    )
    parser.add_argument("--deap-root", type=str, default=None)
    parser.add_argument("--seed-root", type=str, default=None)
    parser.add_argument("--seedv-root", type=str, default=None)
    parser.add_argument("--dreamer-root", type=str, default=None)
    parser.add_argument("--hci-root", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_roots: dict[str, str] = {}
    if args.deap_root:
        data_roots["DEAP"] = args.deap_root
    if args.seed_root:
        data_roots["SEED"] = args.seed_root
    if args.seedv_root:
        data_roots["SEED_V"] = args.seedv_root
    if args.dreamer_root:
        data_roots["DREAMER"] = args.dreamer_root
    if args.hci_root:
        data_roots["MAHNOB_HCI"] = args.hci_root

    all_results: dict[str, dict[str, Any]] = {}
    n_ok = 0
    n_fail = 0

    for exp in EXPERIMENT_CONFIGS:
        try:
            result = _run_single(exp, args.dry_run, output_dir, data_roots)
            all_results[exp["name"]] = result
            n_ok += 1
        except Exception as exc:
            logger.error("Experiment %s FAILED: %s", exp["name"], exc)
            all_results[exp["name"]] = {"error": str(exc)}
            n_fail += 1

    summary_path = output_dir / "results_all.json"
    summary_path.write_text(
        json.dumps(all_results, indent=2, default=_json_default),
        encoding="utf-8",
    )

    logger.info("\n" + "=" * 60)
    logger.info("ALL EXPERIMENTS COMPLETE: %d OK, %d FAILED", n_ok, n_fail)
    logger.info("Results summary: %s", summary_path)
    logger.info("=" * 60)

    for name, res in all_results.items():
        if "error" in res:
            logger.info("  %-40s FAIL: %s", name, res["error"])
        else:
            logger.info(
                "  %-40s acc=%.4f  f1=%.4f",
                name,
                res.get("mean_accuracy", 0),
                res.get("mean_f1_macro", 0),
            )

    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
