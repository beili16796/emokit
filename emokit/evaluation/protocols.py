# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Evaluation protocols: LOSO, subject-dependent, and session-based splits."""

from __future__ import annotations

import copy
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
    X: np.ndarray | dict[str, np.ndarray],
    y: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[Any, np.ndarray, Any, np.ndarray | None]:
    """Split arrays into train and validation sets with stratification.

    Supports both plain arrays and modality dicts.

    Args:
        X: Feature array or dict of modality arrays.
        y: Label array.
        val_fraction: Fraction reserved for validation.
        seed: Random seed.

    Returns:
        ``(X_train, y_train, X_val, y_val)`` tuple.
    """
    if val_fraction <= 0 or len(np.unique(y)) < 2 or len(y) < 4:
        return X, y, None, None  # type: ignore[return-value]

    ref = next(iter(X.values())) if isinstance(X, dict) else X
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    train_idx, val_idx = next(splitter.split(ref, y))

    if isinstance(X, dict):
        return (
            _split_dict(X, train_idx),
            y[train_idx],
            _split_dict(X, val_idx),
            y[val_idx],
        )
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
        output_config: dict[str, Any] | None = None,
    ) -> None:
        self.dataset = dataset
        self.feature_pipeline = feature_pipeline
        self.model_config = model_config
        self.model_name = model_name
        self.seed = seed
        self.val_fraction = val_fraction
        self.output_config = output_config or {}

    def run(self) -> dict[str, Any]:
        """Execute LOSO evaluation over all subjects.

        Supports both unimodal (flat array) and multimodal (dict of arrays)
        models.  When the target model has ``multimodal = True``, per-subject
        data is stored as ``dict[str, np.ndarray]`` keyed by modality name
        so that the model receives the input format it expects.

        Returns:
            Dict with ``per_subject``, ``mean``, ``std``, and ``config``.
        """
        set_seed(self.seed)
        subject_ids = self.dataset.get_subject_ids()
        logger.info("Starting LOSO evaluation over %d subjects", len(subject_ids))

        subject_data: dict[int, dict[str, np.ndarray]] = {}
        for sid in subject_ids:
            raw = self.dataset.read_raw(sid)
            if "labels" not in raw:
                logger.warning("No labels for subject %d, skipping", sid)
                continue
            if "eeg" not in raw and self.model_name not in {"DGCCA-AM"}:
                logger.warning("No data for subject %d, skipping", sid)
                continue
            subject_data[sid] = raw

        per_subject: dict[int, dict[str, Any]] = {}
        per_subject_raw_preds: dict[int, dict[str, list]] = {}

        for test_sid in subject_data:
            logger.info("LOSO fold: test subject = %d", test_sid)

            test_raw = subject_data[test_sid]
            y_test = np.asarray(test_raw["labels"])
            train_raws: list[dict[str, np.ndarray]] = []
            train_ys: list[np.ndarray] = []
            train_sid_arrays: list[np.ndarray] = []
            for sid, raw_s in subject_data.items():
                if sid != test_sid:
                    y_s = np.asarray(raw_s["labels"])
                    train_raws.append(raw_s)
                    train_ys.append(y_s)
                    train_sid_arrays.append(np.full(len(y_s), sid, dtype=np.int64))

            if not train_raws:
                logger.warning(
                    "No training data for fold test_subject=%d "
                    "(only 1 subject?), skipping",
                    test_sid,
                )
                continue

            y_train_all = np.concatenate(train_ys, axis=0)
            train_subject_ids = np.concatenate(train_sid_arrays, axis=0)
            pipeline = _clone_pipeline(self.feature_pipeline)
            X_train_feat, X_test_feat = _prepare_model_features(
                train_raws=train_raws,
                test_raw=test_raw,
                y_train=y_train_all,
                pipeline=pipeline,
                model_name=self.model_name,
                model_config=self.model_config,
            )
            X_tr, y_tr, X_val, y_val = _stratified_val_split_any(
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

            if self.output_config.get("save_checkpoints", False):
                ckpt_dir = (
                    Path(self.output_config.get("results_dir", "results"))
                    / "checkpoints"
                )
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                ckpt_path = ckpt_dir / f"subject_{int(test_sid):02d}_best.pt"
                model.save(str(ckpt_path))

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
                "output": self.output_config,
            },
        )
        result["per_subject_raw_preds"] = per_subject_raw_preds
        if self.output_config.get("save_final_model", False) and subject_data:
            final_model = build_model(self.model_name, self.model_config)
            all_raws = [subject_data[sid] for sid in sorted(subject_data)]
            all_labels = np.concatenate(
                [
                    np.asarray(subject_data[sid]["labels"])
                    for sid in sorted(subject_data)
                ],
                axis=0,
            )
            final_pipeline = _clone_pipeline(self.feature_pipeline)
            X_all, _ = _prepare_model_features(
                train_raws=all_raws,
                test_raw=all_raws[0],
                y_train=all_labels,
                pipeline=final_pipeline,
                model_name=self.model_name,
                model_config=self.model_config,
            )
            final_model.fit(X_all, all_labels)
            final_path = (
                Path(self.output_config.get("results_dir", "results"))
                / "final_model.pt"
            )
            final_model.save(str(final_path))
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


class MultiModelLOSOEvaluator:
    """Run LOSO evaluation for multiple models on one dataset config."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def run(self) -> dict[str, Any]:
        from emokit.datasets import load_dataset

        ds_kwargs: dict[str, Any] = {"root": self.cfg.dataset.root}
        if self.cfg.dataset.subjects is not None:
            ds_kwargs["subjects"] = self.cfg.dataset.subjects
        if self.cfg.dataset.window_sec is not None:
            ds_kwargs["window_sec"] = self.cfg.dataset.window_sec
        if self.cfg.dataset.overlap is not None:
            ds_kwargs["overlap"] = self.cfg.dataset.overlap
        if self.cfg.dataset.modalities is not None:
            ds_kwargs["modalities"] = self.cfg.dataset.modalities
        if (
            self.cfg.dataset.label_axis is not None
            and self.cfg.dataset.name != "SYNTHETIC"
        ):
            ds_kwargs["label_axis"] = self.cfg.dataset.label_axis
        if hasattr(self.cfg.dataset, "params") and self.cfg.dataset.params:
            ds_kwargs.update(self.cfg.dataset.params)
        dataset = load_dataset(self.cfg.dataset.name, **ds_kwargs)

        per_model_results: dict[str, Any] = {}
        for model_spec in self.cfg.models_to_run:
            logger.info("Running multi-model LOSO: %s", model_spec.name)
            evaluator = LOSOEvaluator(
                dataset=dataset,
                feature_pipeline=_build_pipeline_from_config(self.cfg.feature_pipeline),
                model_config={
                    **(self.cfg.model_defaults or {}),
                    **(model_spec.params or {}),
                },
                model_name=model_spec.name,
                seed=self.cfg.experiment.seed,
                val_fraction=self.cfg.evaluation.val_fraction,
                output_config=dict(self.cfg.output.model_dump()),
            )
            result = evaluator.run()
            per_model_results[model_spec.name] = {
                "mean_acc": result.get("mean", {}).get("accuracy", 0.0),
                "std_acc": result.get("std", {}).get("accuracy", 0.0),
                "mean_f1": result.get("mean", {}).get("f1_macro", 0.0),
                "per_subject": result.get("per_subject", {}),
            }

        mean_accs = [item["mean_acc"] for item in per_model_results.values()]
        return {
            "per_model": per_model_results,
            "config": {
                "dataset_name": self.cfg.dataset.name,
                "protocol": self.cfg.evaluation.protocol,
                "seed": self.cfg.experiment.seed,
            },
            "mean": {"accuracy": float(np.mean(mean_accs)) if mean_accs else 0.0},
            "std": {"accuracy": float(np.std(mean_accs)) if mean_accs else 0.0},
        }


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
                f"SessionEvaluator requires at least 2 sessions, got {len(sessions)}"
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


def _peek_model_cls(model_name: str) -> type:
    """Look up a model class without instantiating it."""
    from emokit.models.base import registry as model_registry

    return model_registry[model_name]


def _concat_dicts(dicts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate a list of modality dicts along axis 0."""
    keys = dicts[0].keys()
    return {k: np.concatenate([d[k] for d in dicts], axis=0) for k in keys}


def _split_dict(X: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    """Index into each array of a modality dict."""
    return {k: v[idx] for k, v in X.items()}


def _clone_pipeline(pipeline: FeaturePipeline) -> FeaturePipeline:
    """Deep-copy a pipeline so fitted state is not shared across folds."""
    return copy.deepcopy(pipeline)


def _slice_features(X: Any, indices: np.ndarray) -> Any:
    """Slice numpy- or dict-backed feature containers."""
    if X is None:
        return None
    if isinstance(X, dict):
        return {k: _slice_features(v, indices) for k, v in X.items()}
    return X[indices]


def _stratified_val_split_any(
    X: Any,
    y: np.ndarray,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[Any, np.ndarray, Any, np.ndarray]:
    """Stratified split that supports dict feature containers."""
    if val_fraction <= 0 or len(np.unique(y)) < 2 or len(y) < 4:
        return X, y, None, None  # type: ignore[return-value]

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    try:
        train_idx, val_idx = next(splitter.split(np.arange(len(y)), y))
    except ValueError:
        return X, y, None, None  # type: ignore[return-value]
    return (
        _slice_features(X, train_idx),
        y[train_idx],
        _slice_features(X, val_idx),
        y[val_idx],
    )


def _build_pipeline_from_config(cfg: Any) -> FeaturePipeline:
    """Instantiate a fresh feature pipeline from config."""
    from emokit.features.base import GLOBAL_REGISTRY as TRANSFORM_REGISTRY

    steps: list[tuple[str, Any]] = []
    for step_cfg in cfg.steps:
        transform_cls = TRANSFORM_REGISTRY[step_cfg.name]
        steps.append((step_cfg.name, transform_cls(**(step_cfg.params or {}))))
    return FeaturePipeline(steps)


def _concat_raw(train_raws: list[dict[str, np.ndarray]], key: str) -> np.ndarray | None:
    arrays = [
        np.asarray(raw[key])
        for raw in train_raws
        if key in raw and raw[key] is not None
    ]
    if not arrays:
        return None
    return np.concatenate(arrays, axis=0)


def _extract_gsr_features(
    train_raws: list[dict[str, np.ndarray]],
    test_raw: dict[str, np.ndarray],
    target_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    from emokit.features import GSRExtractor

    train_gsr = _concat_raw(train_raws, "gsr")
    test_gsr = (
        np.asarray(test_raw["gsr"])
        if "gsr" in test_raw and test_raw["gsr"] is not None
        else None
    )
    extractor = GSRExtractor()
    if train_gsr is None:
        n_train = sum(len(np.asarray(raw["labels"])) for raw in train_raws)
        n_test = len(np.asarray(test_raw["labels"]))
        return np.zeros((n_train, target_dim), dtype=np.float32), np.zeros(
            (n_test, target_dim), dtype=np.float32
        )
    train_feat = extractor.transform(train_gsr)
    test_feat = (
        extractor.transform(test_gsr)
        if test_gsr is not None
        else np.zeros(
            (len(np.asarray(test_raw["labels"])), target_dim), dtype=np.float32
        )
    )
    return _pad_features(train_feat, target_dim), _pad_features(test_feat, target_dim)


def _extract_ecg_features(
    train_raws: list[dict[str, np.ndarray]],
    test_raw: dict[str, np.ndarray],
    target_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    from emokit.features import HRVExtractor

    train_ecg = _concat_raw(train_raws, "ecg")
    test_ecg = (
        np.asarray(test_raw["ecg"])
        if "ecg" in test_raw and test_raw["ecg"] is not None
        else None
    )
    extractor = HRVExtractor()
    if train_ecg is None:
        n_train = sum(len(np.asarray(raw["labels"])) for raw in train_raws)
        n_test = len(np.asarray(test_raw["labels"]))
        return np.zeros((n_train, target_dim), dtype=np.float32), np.zeros(
            (n_test, target_dim), dtype=np.float32
        )
    train_feat = extractor.transform(train_ecg)
    test_feat = (
        extractor.transform(test_ecg)
        if test_ecg is not None
        else np.zeros(
            (len(np.asarray(test_raw["labels"])), target_dim), dtype=np.float32
        )
    )
    return _pad_features(train_feat, target_dim), _pad_features(test_feat, target_dim)


def _pad_features(X: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad or trim 2-D features to the required width."""
    arr = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] == target_dim:
        return arr
    if arr.shape[1] > target_dim:
        return arr[:, :target_dim]
    pad = np.zeros((arr.shape[0], target_dim - arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def _prepare_model_features(
    train_raws: list[dict[str, np.ndarray]],
    test_raw: dict[str, np.ndarray],
    y_train: np.ndarray,
    pipeline: FeaturePipeline,
    model_name: str,
    model_config: dict[str, Any],
) -> tuple[Any, Any]:
    """Build model-specific train/test inputs from raw modality arrays."""
    train_eeg = _concat_raw(train_raws, "eeg")
    test_eeg = (
        np.asarray(test_raw["eeg"])
        if "eeg" in test_raw and test_raw["eeg"] is not None
        else None
    )

    eeg_train_feat = None
    eeg_test_feat = None
    if train_eeg is not None and test_eeg is not None:
        eeg_train_feat = pipeline.fit_transform(train_eeg, y_train)
        eeg_test_feat = pipeline.transform(test_eeg)

    if model_name == "CNN-LSTM":
        if model_config.get("input_type", "de") == "raw":
            return train_eeg, test_eeg
        assert eeg_train_feat is not None and eeg_test_feat is not None
        return (
            eeg_train_feat.reshape(eeg_train_feat.shape[0], -1),
            eeg_test_feat.reshape(eeg_test_feat.shape[0], -1),
        )

    if model_name == "DGCNN":
        return eeg_train_feat, eeg_test_feat

    if model_name == "PR-PL":
        assert eeg_train_feat is not None and eeg_test_feat is not None
        return (
            eeg_train_feat.reshape(eeg_train_feat.shape[0], -1),
            eeg_test_feat.reshape(eeg_test_feat.shape[0], -1),
        )

    if model_name == "BiDAE":
        assert eeg_train_feat is not None and eeg_test_feat is not None
        mod2_dim = int(model_config.get("n_feat2", model_config.get("n_feat_mod2", 3)))
        gsr_train, gsr_test = _extract_gsr_features(train_raws, test_raw, mod2_dim)
        return (
            {
                "mod1": eeg_train_feat.reshape(eeg_train_feat.shape[0], -1),
                "mod2": gsr_train,
            },
            {
                "mod1": eeg_test_feat.reshape(eeg_test_feat.shape[0], -1),
                "mod2": gsr_test,
            },
        )

    if model_name == "DGCCA-AM":
        assert eeg_train_feat is not None and eeg_test_feat is not None
        gsr_dim = int(model_config.get("n_feat_gsr", 3))
        ecg_dim = int(model_config.get("n_feat_ecg", 5))
        gsr_train, gsr_test = _extract_gsr_features(train_raws, test_raw, gsr_dim)
        ecg_train, ecg_test = _extract_ecg_features(train_raws, test_raw, ecg_dim)
        return (
            {
                "eeg": eeg_train_feat.reshape(eeg_train_feat.shape[0], -1),
                "gsr": gsr_train,
                "ecg": ecg_train,
            },
            {
                "eeg": eeg_test_feat.reshape(eeg_test_feat.shape[0], -1),
                "gsr": gsr_test,
                "ecg": ecg_test,
            },
        )

    if model_name == "Transformer-MM":
        assert eeg_train_feat is not None and eeg_test_feat is not None
        periph_dim = int(model_config.get("n_peripheral_feat", 7))
        gsr_train, gsr_test = _extract_gsr_features(
            train_raws, test_raw, min(3, periph_dim)
        )
        ecg_train, ecg_test = _extract_ecg_features(
            train_raws, test_raw, max(periph_dim - gsr_train.shape[1], 0)
        )
        periph_train = _pad_features(
            np.concatenate([gsr_train, ecg_train], axis=1), periph_dim
        )
        periph_test = _pad_features(
            np.concatenate([gsr_test, ecg_test], axis=1), periph_dim
        )
        return (
            {"eeg": eeg_train_feat, "peripheral": periph_train},
            {"eeg": eeg_test_feat, "peripheral": periph_test},
        )

    return eeg_train_feat, eeg_test_feat


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
    set_seed(cfg.experiment.seed)
    if getattr(cfg, "models_to_run", None):
        return MultiModelLOSOEvaluator(cfg).run()

    if cfg.model is None:
        raise EmoKitConfigError("Single-model config requires a 'model' section.")

    from emokit.datasets import load_dataset

    ds_kwargs: dict[str, Any] = {"root": cfg.dataset.root}
    if cfg.dataset.subjects is not None:
        ds_kwargs["subjects"] = cfg.dataset.subjects
    if cfg.dataset.window_sec is not None:
        ds_kwargs["window_sec"] = cfg.dataset.window_sec
    if cfg.dataset.overlap is not None:
        ds_kwargs["overlap"] = cfg.dataset.overlap
    if cfg.dataset.modalities is not None:
        ds_kwargs["modalities"] = cfg.dataset.modalities
    if cfg.dataset.label_axis is not None and cfg.dataset.name != "SYNTHETIC":
        ds_kwargs["label_axis"] = cfg.dataset.label_axis
    if hasattr(cfg.dataset, "params") and cfg.dataset.params:
        ds_kwargs.update(cfg.dataset.params)
    dataset = load_dataset(cfg.dataset.name, **ds_kwargs)

    evaluator = protocol_cls(
        dataset=dataset,
        feature_pipeline=_build_pipeline_from_config(cfg.feature_pipeline),
        model_config=cfg.model.params or {},
        model_name=cfg.model.name,
        seed=cfg.experiment.seed,
        val_fraction=cfg.evaluation.val_fraction,
        output_config=dict(cfg.output.model_dump()),
    )
    return evaluator.run()


class CrossCorpusEvaluator:
    """Cross-corpus generalisation evaluator.

    Trains on *all* subjects of ``source_dataset`` and evaluates per-subject
    on ``target_dataset``.  When the two corpora have different EEG montages,
    channels are automatically aligned using the International 10-20 system.

    This protocol is essential for validating domain generalisation claims
    in affective computing papers (e.g. training on SEED → testing on DREAMER).

    Args:
        source_dataset: Dataset to train on (all subjects).
        target_dataset: Dataset to evaluate on (per-subject metrics).
        feature_pipeline: Shared feature extraction pipeline.
        model_config: Config dict forwarded to the model constructor.
        model_name: Registered model name.
        seed: Global random seed.
        val_fraction: Fraction of source data held out for validation.
    """

    def __init__(
        self,
        source_dataset: BaseDataset,
        target_dataset: BaseDataset,
        feature_pipeline: FeaturePipeline,
        model_config: dict[str, Any],
        model_name: str,
        seed: int = 42,
        val_fraction: float = 0.1,
    ) -> None:
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.feature_pipeline = feature_pipeline
        self.model_config = model_config
        self.model_name = model_name
        self.seed = seed
        self.val_fraction = val_fraction

    def _align_if_needed(
        self, X: np.ndarray, src_names: list[str], tgt_names: list[str]
    ) -> np.ndarray:
        """Subset source channels to match target montage if different."""
        if len(src_names) == len(tgt_names):
            return X
        from emokit.features.channel_align import subset_features

        return subset_features(X, src_names, tgt_names)

    def run(self) -> dict[str, Any]:
        """Train on source corpus, evaluate per-subject on target.

        Returns:
            Dict with ``per_subject``, ``mean``, ``std``, ``config``.
        """
        set_seed(self.seed)
        src_ch = self.source_dataset.get_channel_names("eeg")
        tgt_ch = self.target_dataset.get_channel_names("eeg")
        do_align = len(src_ch) != len(tgt_ch)

        if do_align:
            logger.info(
                "Channel alignment: %d-ch source → %d-ch target",
                len(src_ch),
                len(tgt_ch),
            )

        # --- Collect source data ---
        src_ids = self.source_dataset.get_subject_ids()
        src_Xs: list[np.ndarray] = []
        src_ys: list[np.ndarray] = []
        for sid in src_ids:
            raw = self.source_dataset.read_raw(sid)
            eeg = raw.get("eeg")
            if eeg is None or "labels" not in raw:
                continue
            if eeg.ndim == 3:
                eeg = eeg.reshape(eeg.shape[0], -1)
            src_Xs.append(eeg)
            src_ys.append(np.asarray(raw["labels"]))

        if not src_Xs:
            raise EmoKitConfigError("No source data loaded")

        X_src = np.concatenate(src_Xs, axis=0)
        y_src = np.concatenate(src_ys, axis=0)

        # --- Collect target data (per-subject) ---
        tgt_ids = self.target_dataset.get_subject_ids()
        tgt_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for sid in tgt_ids:
            raw = self.target_dataset.read_raw(sid)
            eeg = raw.get("eeg")
            if eeg is None or "labels" not in raw:
                continue
            if eeg.ndim == 3:
                eeg = eeg.reshape(eeg.shape[0], -1)
            tgt_data[sid] = (eeg, np.asarray(raw["labels"]))

        # --- Feature pipeline ---
        pipeline = _clone_pipeline(self.feature_pipeline)
        X_src_feat = pipeline.fit_transform(X_src, y_src)

        # Channel alignment on feature space
        if do_align and X_src_feat.ndim >= 2:
            X_src_feat = self._align_if_needed(X_src_feat, src_ch, tgt_ch)

        # --- Train/val split ---
        X_tr, y_tr, X_val, y_val = _stratified_val_split_any(
            X_src_feat, y_src, self.val_fraction, self.seed
        )

        model = build_model(self.model_name, self.model_config)
        model.fit(X_tr, y_tr, X_val, y_val)

        # --- Evaluate per target subject ---
        per_subject: dict[int, dict[str, Any]] = {}
        for sid, (X_tgt_raw, y_tgt) in tgt_data.items():
            X_tgt_feat = pipeline.transform(X_tgt_raw)
            if do_align and X_tgt_feat.ndim >= 2:
                X_tgt_feat = self._align_if_needed(X_tgt_feat, tgt_ch, tgt_ch)
            y_pred = model.predict(X_tgt_feat)
            metrics = compute_metrics(y_tgt, y_pred)
            per_subject[sid] = metrics
            logger.info(
                "Target subject %d — acc=%.4f f1=%.4f",
                sid,
                metrics["accuracy"],
                metrics["f1_macro"],
            )

        return _aggregate_results(
            per_subject,
            config={
                "source_dataset": type(self.source_dataset).__name__,
                "target_dataset": type(self.target_dataset).__name__,
                "model_name": self.model_name,
                "seed": self.seed,
                "protocol": "cross_corpus",
                "channel_alignment": do_align,
                "source_channels": len(src_ch),
                "target_channels": len(tgt_ch),
            },
        )


def _json_default(obj: Any) -> Any:
    """JSON serialiser fallback for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
