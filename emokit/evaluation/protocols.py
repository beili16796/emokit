# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Evaluation protocols: LOSO, subject-dependent, and session-based splits."""

from __future__ import annotations

import csv
import inspect
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from emokit.datasets.base import BaseDataset
from emokit.features.base import FeaturePipeline
from emokit.models.base import build_model
from emokit.utils import EmoKitConfigError, set_seed

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute standard classification metrics.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        y_proba: Optional predicted class probabilities (unused, reserved
            for future ROC-AUC support).

    Returns:
        Dict with keys ``accuracy``, ``f1_macro``, ``f1_weighted``, and
        ``confusion_matrix``.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _stratified_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train and validation sets with stratification.

    Args:
        X: Feature array.
        y: Label array.
        val_fraction: Fraction reserved for validation.
        seed: Random seed.

    Returns:
        ``(X_train, y_train, X_val, y_val)`` tuple.
    """
    if val_fraction <= 0 or len(np.unique(y)) < 2 or len(y) < 4:
        return X, y, None, None  # type: ignore[return-value]

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    train_idx, val_idx = next(splitter.split(X, y))
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


class LOSOEvaluator:
    """Leave-One-Subject-Out cross-validation evaluator.

    Args:
        dataset: Loaded :class:`BaseDataset` instance.
        feature_pipeline: :class:`FeaturePipeline` for feature extraction.
        model_config: Config dict forwarded to the model constructor.
        model_name: Registered model name.
        seed: Global random seed.
        val_fraction: Fraction of training data held out for validation.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        feature_pipeline: FeaturePipeline,
        model_config: dict[str, Any],
        model_name: str,
        seed: int = 42,
        val_fraction: float = 0.1,
    ) -> None:
        self.dataset = dataset
        self.feature_pipeline = feature_pipeline
        self.model_config = model_config
        self.model_name = model_name
        self.seed = seed
        self.val_fraction = val_fraction

    def run(self) -> dict[str, Any]:
        """Execute LOSO evaluation over all subjects.

        Returns:
            Dict with ``per_subject``, ``mean``, ``std``, and ``config``.
        """
        set_seed(self.seed)
        subject_ids = self.dataset.get_subject_ids()
        logger.info("Starting LOSO evaluation over %d subjects", len(subject_ids))

        subject_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for sid in subject_ids:
            raw = self.dataset.read_raw(sid)
            modalities = self.dataset.modalities or [k for k in raw if k != "labels"]
            arrays = [raw[m] for m in modalities if m in raw and m != "labels"]
            if not arrays:
                logger.warning("No data for subject %d, skipping", sid)
                continue
            X_subj = np.concatenate(arrays, axis=1)
            subject_data[sid] = (X_subj, raw["labels"])

        per_subject: dict[int, dict[str, Any]] = {}
        per_subject_raw_preds: dict[int, dict[str, list]] = {}

        for test_sid in subject_data:
            logger.info("LOSO fold: test subject = %d", test_sid)

            X_test_raw, y_test = subject_data[test_sid]
            train_Xs: list[np.ndarray] = []
            train_ys: list[np.ndarray] = []
            train_sid_arrays: list[np.ndarray] = []
            for sid, (X_s, y_s) in subject_data.items():
                if sid != test_sid:
                    train_Xs.append(X_s)
                    train_ys.append(y_s)
                    train_sid_arrays.append(np.full(len(y_s), sid, dtype=np.int64))

            if not train_Xs:
                logger.warning(
                    "No training data for fold test_subject=%d "
                    "(only 1 subject?), skipping",
                    test_sid,
                )
                continue

            X_train_raw = np.concatenate(train_Xs, axis=0)
            y_train_all = np.concatenate(train_ys, axis=0)
            train_subject_ids = np.concatenate(train_sid_arrays, axis=0)

            pipeline = _clone_pipeline(self.feature_pipeline)
            X_train_feat = pipeline.fit_transform(X_train_raw, y_train_all)
            X_test_feat = pipeline.transform(X_test_raw)

            X_tr, y_tr, X_val, y_val = _stratified_val_split(
                X_train_feat, y_train_all, self.val_fraction, self.seed
            )

            model = build_model(self.model_name, self.model_config)

            fit_params = inspect.signature(model.fit).parameters
            fit_kwargs: dict[str, Any] = {}
            if "X_target" in fit_params:
                fit_kwargs["X_target"] = X_test_feat
            if "subject_ids" in fit_params:
                fit_kwargs["subject_ids"] = train_subject_ids

            if fit_kwargs:
                model.fit(X_tr, y_tr, **fit_kwargs)
            else:
                model.fit(X_tr, y_tr, X_val, y_val)

            y_pred = model.predict(X_test_feat)
            metrics = compute_metrics(y_test, y_pred)
            per_subject[test_sid] = metrics
            per_subject_raw_preds[test_sid] = {
                "y_true": y_test.tolist(),
                "y_pred": y_pred.tolist(),
            }
            logger.info(
                "Subject %d — acc=%.4f  f1=%.4f",
                test_sid,
                metrics["accuracy"],
                metrics["f1_macro"],
            )

        result = _aggregate_results(
            per_subject,
            config={
                "dataset": type(self.dataset).__name__,
                "dataset_name": type(self.dataset).__name__,
                "model": self.model_name,
                "model_name": self.model_name,
                "feature_pipeline_str": str(
                    [(n, type(t).__name__) for n, t in self.feature_pipeline.steps]
                ),
                "seed": self.seed,
                "protocol": "loso",
            },
        )
        result["per_subject_raw_preds"] = per_subject_raw_preds
        return result

    @classmethod
    def run_from_yaml(cls, yaml_path: str) -> dict[str, Any]:
        """Load a YAML config and run the full LOSO evaluation.

        Args:
            yaml_path: Path to experiment YAML file.

        Returns:
            Evaluation results dict.
        """
        from emokit.evaluation.config import ConfigLoader

        cfg = ConfigLoader.load(yaml_path)
        return _run_from_full_config(cfg, protocol_cls=cls)

    @classmethod
    def run_from_config(cls, cfg: dict | Any) -> dict[str, Any]:
        """Run LOSO evaluation from a config dict or FullConfig object.

        Args:
            cfg: Either a FullConfig pydantic object or a plain dict.

        Returns:
            Evaluation results dict.
        """
        if isinstance(cfg, dict):
            from emokit.evaluation.config import FullConfig

            cfg = FullConfig(**cfg)
        return _run_from_full_config(cfg, protocol_cls=cls)


class SubjectDependentEvaluator:
    """Within-subject train/test evaluation (average across subjects).

    Args:
        dataset: Loaded :class:`BaseDataset` instance.
        feature_pipeline: :class:`FeaturePipeline` for feature extraction.
        model_config: Config dict forwarded to the model constructor.
        model_name: Registered model name.
        seed: Global random seed.
        val_fraction: Fraction of training data held out for validation.
        test_fraction: Fraction of each subject's data used for testing.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        feature_pipeline: FeaturePipeline,
        model_config: dict[str, Any],
        model_name: str,
        seed: int = 42,
        val_fraction: float = 0.1,
        test_fraction: float = 0.2,
    ) -> None:
        self.dataset = dataset
        self.feature_pipeline = feature_pipeline
        self.model_config = model_config
        self.model_name = model_name
        self.seed = seed
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction

    def run(self) -> dict[str, Any]:
        """Execute subject-dependent evaluation.

        Returns:
            Dict with ``per_subject``, ``mean``, ``std``, and ``config``.
        """
        set_seed(self.seed)
        subject_ids = self.dataset.get_subject_ids()
        logger.info(
            "Starting subject-dependent evaluation over %d subjects",
            len(subject_ids),
        )

        per_subject: dict[int, dict[str, Any]] = {}

        for sid in subject_ids:
            raw = self.dataset.read_raw(sid)
            modalities = self.dataset.modalities or [k for k in raw if k != "labels"]
            arrays = [raw[m] for m in modalities if m in raw and m != "labels"]
            if not arrays:
                logger.warning("No data for subject %d, skipping", sid)
                continue

            X_all = np.concatenate(arrays, axis=1)
            y_all = raw["labels"]

            if len(y_all) < 4 or len(np.unique(y_all)) < 2:
                logger.warning("Insufficient data for subject %d, skipping", sid)
                continue

            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=self.test_fraction, random_state=self.seed
            )
            train_idx, test_idx = next(splitter.split(X_all, y_all))

            X_train_raw, y_train_all = X_all[train_idx], y_all[train_idx]
            X_test_raw, y_test = X_all[test_idx], y_all[test_idx]

            pipeline = _clone_pipeline(self.feature_pipeline)
            X_train_feat = pipeline.fit_transform(X_train_raw, y_train_all)
            X_test_feat = pipeline.transform(X_test_raw)

            X_tr, y_tr, X_val, y_val = _stratified_val_split(
                X_train_feat, y_train_all, self.val_fraction, self.seed
            )

            model = build_model(self.model_name, self.model_config)
            model.fit(X_tr, y_tr, X_val, y_val)
            metrics = model.evaluate(X_test_feat, y_test)
            per_subject[sid] = metrics
            logger.info(
                "Subject %d — acc=%.4f  f1=%.4f",
                sid,
                metrics["accuracy"],
                metrics["f1_macro"],
            )

        return _aggregate_results(
            per_subject,
            config={
                "dataset_name": type(self.dataset).__name__,
                "model_name": self.model_name,
                "feature_pipeline_str": str(
                    [(n, type(t).__name__) for n, t in self.feature_pipeline.steps]
                ),
                "seed": self.seed,
                "protocol": "subject_dependent",
            },
        )

    @classmethod
    def run_from_yaml(cls, yaml_path: str) -> dict[str, Any]:
        """Load a YAML config and run the full subject-dependent evaluation.

        Args:
            yaml_path: Path to experiment YAML file.

        Returns:
            Evaluation results dict.
        """
        from emokit.evaluation.config import ConfigLoader

        cfg = ConfigLoader.load(yaml_path)
        return _run_from_full_config(cfg, protocol_cls=cls)


class SessionEvaluator:
    """Cross-session evaluator: train on sessions 1..N-1, test on session N.

    Requires a multi-session dataset whose ``read_raw`` returns data
    segmented by session (with a ``sessions`` attribute).

    Args:
        dataset: Loaded :class:`BaseDataset` instance with session support.
        feature_pipeline: :class:`FeaturePipeline` for feature extraction.
        model_config: Config dict forwarded to the model constructor.
        model_name: Registered model name.
        seed: Global random seed.
        val_fraction: Fraction of training data held out for validation.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        feature_pipeline: FeaturePipeline,
        model_config: dict[str, Any],
        model_name: str,
        seed: int = 42,
        val_fraction: float = 0.1,
    ) -> None:
        self.dataset = dataset
        self.feature_pipeline = feature_pipeline
        self.model_config = model_config
        self.model_name = model_name
        self.seed = seed
        self.val_fraction = val_fraction

    def run(self) -> dict[str, Any]:
        """Execute session-based evaluation per subject.

        For each subject, trains on sessions ``1..N-1`` and tests on
        session ``N``.

        Returns:
            Dict with ``per_subject``, ``mean``, ``std``, and ``config``.
        """
        set_seed(self.seed)
        subject_ids = self.dataset.get_subject_ids()
        sessions: list[int] = getattr(self.dataset, "sessions", [1, 2, 3])

        if len(sessions) < 2:
            raise EmoKitConfigError(
                "SessionEvaluator requires at least 2 sessions, " f"got {len(sessions)}"
            )

        train_sessions = sessions[:-1]
        test_session = sessions[-1]
        logger.info(
            "Session evaluation: train=%s, test=%s over %d subjects",
            train_sessions,
            test_session,
            len(subject_ids),
        )

        per_subject: dict[int, dict[str, Any]] = {}

        for sid in subject_ids:
            train_Xs: list[np.ndarray] = []
            train_ys: list[np.ndarray] = []
            test_Xs: list[np.ndarray] = []
            test_ys: list[np.ndarray] = []

            original_sessions = getattr(self.dataset, "sessions", None)

            for sess in sessions:
                if hasattr(self.dataset, "sessions"):
                    self.dataset.sessions = [sess]
                raw = self.dataset.read_raw(sid)
                modalities = self.dataset.modalities or [
                    k for k in raw if k != "labels"
                ]
                arrays = [raw[m] for m in modalities if m in raw and m != "labels"]
                if not arrays:
                    continue
                X_s = np.concatenate(arrays, axis=1)
                y_s = raw["labels"]

                if sess == test_session:
                    test_Xs.append(X_s)
                    test_ys.append(y_s)
                else:
                    train_Xs.append(X_s)
                    train_ys.append(y_s)

            if hasattr(self.dataset, "sessions") and original_sessions is not None:
                self.dataset.sessions = original_sessions

            if not train_Xs or not test_Xs:
                logger.warning(
                    "Insufficient session data for subject %d, skipping",
                    sid,
                )
                continue

            X_train_raw = np.concatenate(train_Xs, axis=0)
            y_train_all = np.concatenate(train_ys, axis=0)
            X_test_raw = np.concatenate(test_Xs, axis=0)
            y_test = np.concatenate(test_ys, axis=0)

            pipeline = _clone_pipeline(self.feature_pipeline)
            X_train_feat = pipeline.fit_transform(X_train_raw, y_train_all)
            X_test_feat = pipeline.transform(X_test_raw)

            X_tr, y_tr, X_val, y_val = _stratified_val_split(
                X_train_feat, y_train_all, self.val_fraction, self.seed
            )

            model = build_model(self.model_name, self.model_config)
            model.fit(X_tr, y_tr, X_val, y_val)
            metrics = model.evaluate(X_test_feat, y_test)
            per_subject[sid] = metrics
            logger.info(
                "Subject %d — acc=%.4f  f1=%.4f",
                sid,
                metrics["accuracy"],
                metrics["f1_macro"],
            )

        return _aggregate_results(
            per_subject,
            config={
                "dataset_name": type(self.dataset).__name__,
                "model_name": self.model_name,
                "feature_pipeline_str": str(
                    [(n, type(t).__name__) for n, t in self.feature_pipeline.steps]
                ),
                "seed": self.seed,
                "protocol": "session",
            },
        )

    @classmethod
    def run_from_yaml(cls, yaml_path: str) -> dict[str, Any]:
        """Load a YAML config and run the full session-based evaluation.

        Args:
            yaml_path: Path to experiment YAML file.

        Returns:
            Evaluation results dict.
        """
        from emokit.evaluation.config import ConfigLoader

        cfg = ConfigLoader.load(yaml_path)
        return _run_from_full_config(cfg, protocol_cls=cls)


class ResultLogger:
    """Persist evaluation results to JSON, CSV, and a cumulative database file.

    Args:
        results_dir: Directory where results are saved.
    """

    def __init__(self, results_dir: str = "results/") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def log(self, results: dict[str, Any]) -> Path:
        """Save evaluation results.

        Writes a timestamped JSON file, a summary CSV, and appends a row
        to the cumulative ``results_db.csv`` file.

        Args:
            results: Evaluation results dict (as returned by evaluators).

        Returns:
            Path to the saved JSON file.
        """
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        config = results.get("config", {})
        name = config.get("dataset_name", "unknown")
        protocol = config.get("protocol", "unknown")
        stem = f"{name}_{protocol}_{timestamp}"

        json_path = self.results_dir / f"{stem}.json"
        csv_path = self.results_dir / f"{stem}.csv"
        db_path = self.results_dir / "results_db.csv"

        json_path.write_text(
            json.dumps(results, indent=2, default=_json_default), encoding="utf-8"
        )
        logger.info("Results JSON saved to %s", json_path)

        per_subject = results.get("per_subject", {})
        if per_subject:
            first_metrics = next(iter(per_subject.values()))
            fieldnames = ["subject_id", *sorted(first_metrics.keys())]
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for sid, metrics in sorted(per_subject.items(), key=lambda x: x[0]):
                    row = {"subject_id": sid, **metrics}
                    writer.writerow(row)
            logger.info("Per-subject CSV saved to %s", csv_path)

        mean = results.get("mean", {})
        db_row = {
            "timestamp": timestamp,
            "dataset": config.get("dataset_name", ""),
            "model": config.get("model_name", ""),
            "protocol": protocol,
            "seed": config.get("seed", ""),
            **{f"mean_{k}": v for k, v in mean.items()},
        }
        db_exists = db_path.exists()
        with open(db_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(db_row.keys()))
            if not db_exists:
                writer.writeheader()
            writer.writerow(db_row)
        logger.info("Appended to results DB at %s", db_path)

        return json_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _clone_pipeline(pipeline: FeaturePipeline) -> FeaturePipeline:
    """Deep-copy a pipeline so fitted state is not shared across folds."""
    import copy

    return copy.deepcopy(pipeline)


def _aggregate_results(
    per_subject: dict[int, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Compute mean/std summaries from per-subject metrics."""
    if not per_subject:
        return {"per_subject": {}, "mean": {}, "std": {}, "config": config}

    first_subject_metrics = next(iter(per_subject.values()))
    metric_keys = [k for k in first_subject_metrics if k != "confusion_matrix"]
    values = {k: [m[k] for m in per_subject.values()] for k in metric_keys}

    return {
        "per_subject": per_subject,
        "mean": {k: float(np.mean(v)) for k, v in values.items()},
        "std": {k: float(np.std(v)) for k, v in values.items()},
        "config": config,
    }


def _run_from_full_config(cfg: Any, protocol_cls: type) -> dict[str, Any]:
    """Instantiate components from a :class:`FullConfig` and run evaluation."""
    from emokit.datasets import load_dataset
    from emokit.features.base import GLOBAL_REGISTRY as TRANSFORM_REGISTRY

    set_seed(cfg.experiment.seed)

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
    if hasattr(cfg.dataset, "params") and cfg.dataset.params:
        ds_kwargs.update(cfg.dataset.params)
    dataset = load_dataset(cfg.dataset.name, **ds_kwargs)

    steps: list[tuple[str, Any]] = []
    for step_cfg in cfg.feature_pipeline.steps:
        transform_cls = TRANSFORM_REGISTRY[step_cfg.name]
        steps.append((step_cfg.name, transform_cls(**(step_cfg.params or {}))))
    pipeline = FeaturePipeline(steps)

    evaluator = protocol_cls(
        dataset=dataset,
        feature_pipeline=pipeline,
        model_config=cfg.model.params or {},
        model_name=cfg.model.name,
        seed=cfg.experiment.seed,
        val_fraction=cfg.evaluation.val_fraction,
    )
    return evaluator.run()


def _json_default(obj: Any) -> Any:
    """JSON serialiser fallback for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
