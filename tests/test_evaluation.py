# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Unit tests for the evaluation layer, config loading, and result logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from emokit.datasets.base import BaseDataset
from emokit.evaluation.config import (
    ConfigLoader,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    FeatureStepConfig,
    FullConfig,
    ModelConfig,
)
from emokit.evaluation.protocols import (
    LOSOEvaluator,
    ResultLogger,
    SubjectDependentEvaluator,
    compute_metrics,
)
from emokit.features.base import BaseTransform, FeaturePipeline
from emokit.models.base import BaseModel, registry
from emokit.utils import EmoKitConfigError, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock dataset: 5 subjects, 20 trials each, 2 classes, 10-feature vectors
# ---------------------------------------------------------------------------

class SyntheticDataset(BaseDataset):
    """Tiny synthetic dataset for testing evaluation protocols."""

    def __init__(
        self,
        n_subjects: int = 5,
        n_trials_per_subject: int = 20,
        n_features: int = 10,
        n_classes: int = 2,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(root="/tmp/synthetic", **kwargs)
        self.n_subjects = n_subjects
        self.n_trials_per_subject = n_trials_per_subject
        self.n_features = n_features
        self.n_classes = n_classes
        self._seed = seed
        self._generate()

    def _generate(self) -> None:
        rng = np.random.RandomState(self._seed)
        self._data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for sid in range(1, self.n_subjects + 1):
            X = rng.randn(self.n_trials_per_subject, 1, self.n_features)
            y = rng.randint(0, self.n_classes, size=self.n_trials_per_subject)
            self._data[sid] = (X, y)

    def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
        X, y = self._data[subject_id]
        return {"eeg": X, "labels": y}

    def get_subject_ids(self) -> list[int]:
        return list(range(1, self.n_subjects + 1))

    def get_channel_names(self, modality: str) -> list[str]:
        return [f"ch{i}" for i in range(self.n_features)]

    def get_label_names(self) -> list[str]:
        return [f"class_{i}" for i in range(self.n_classes)]

    def _get_fs(self) -> float:
        return 128.0


# ---------------------------------------------------------------------------
# Sklearn LogisticRegression wrapped as BaseModel
# ---------------------------------------------------------------------------

@registry.register("_TestLogReg")
class LogRegModel(BaseModel):
    """Logistic regression wrapper for testing."""

    def __init__(self, n_classes: int = 2, **kwargs: Any) -> None:
        super().__init__(n_classes=n_classes, device="cpu")
        self._clf = LogisticRegression(max_iter=200, solver="lbfgs")

    def fit(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_val: Any | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        X = np.asarray(X_train)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        self._clf.fit(X, y_train)
        return {"train_loss": [0.0]}

    def predict(self, X: Any) -> np.ndarray:
        arr = np.asarray(X)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return self._clf.predict(arr)

    def predict_proba(self, X: Any) -> np.ndarray:
        arr = np.asarray(X)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return self._clf.predict_proba(arr)


# ---------------------------------------------------------------------------
# Identity transform for pipeline
# ---------------------------------------------------------------------------

class IdentityTransform(BaseTransform):
    """Pass-through transform."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> IdentityTransform:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


# ---------------------------------------------------------------------------
# Tests: compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    """Tests for the ``compute_metrics`` function."""

    def test_perfect_predictions(self) -> None:
        y = np.array([0, 0, 1, 1, 0, 1])
        metrics = compute_metrics(y, y)
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_macro"] == pytest.approx(1.0)
        assert metrics["f1_weighted"] == pytest.approx(1.0)
        assert isinstance(metrics["confusion_matrix"], list)

    def test_random_predictions(self) -> None:
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 3, size=100)
        y_pred = rng.randint(0, 3, size=100)
        metrics = compute_metrics(y_true, y_pred)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1_macro"] <= 1.0
        assert 0.0 <= metrics["f1_weighted"] <= 1.0

    def test_with_proba(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.1, 0.9]])
        metrics = compute_metrics(y_true, y_pred, y_proba)
        assert "accuracy" in metrics

    def test_single_class(self) -> None:
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == pytest.approx(1.0)

    def test_confusion_matrix_shape(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 1, 2])
        metrics = compute_metrics(y_true, y_pred)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)


# ---------------------------------------------------------------------------
# Tests: LOSO evaluator
# ---------------------------------------------------------------------------


class TestLOSOEvaluator:
    """Tests for the ``LOSOEvaluator``."""

    def test_full_run(self) -> None:
        set_seed(42)
        ds = SyntheticDataset(n_subjects=5, n_trials_per_subject=20, seed=42)
        pipeline = FeaturePipeline([("identity", IdentityTransform())])
        evaluator = LOSOEvaluator(
            dataset=ds,
            feature_pipeline=pipeline,
            model_config={"n_classes": 2},
            model_name="_TestLogReg",
            seed=42,
        )
        results = evaluator.run()

        assert "per_subject" in results
        assert "mean" in results
        assert "std" in results
        assert "config" in results
        assert len(results["per_subject"]) == 5
        assert "accuracy" in results["mean"]
        assert "f1_macro" in results["mean"]

    def test_output_keys_per_subject(self) -> None:
        ds = SyntheticDataset(n_subjects=3, n_trials_per_subject=30, seed=7)
        pipeline = FeaturePipeline([("id", IdentityTransform())])
        evaluator = LOSOEvaluator(
            dataset=ds,
            feature_pipeline=pipeline,
            model_config={"n_classes": 2},
            model_name="_TestLogReg",
            seed=7,
        )
        results = evaluator.run()
        for sid, m in results["per_subject"].items():
            assert "accuracy" in m
            assert "f1_macro" in m
            assert "f1_weighted" in m

    def test_config_in_results(self) -> None:
        ds = SyntheticDataset(n_subjects=2, n_trials_per_subject=20, seed=1)
        pipeline = FeaturePipeline([("id", IdentityTransform())])
        evaluator = LOSOEvaluator(
            dataset=ds,
            feature_pipeline=pipeline,
            model_config={"n_classes": 2},
            model_name="_TestLogReg",
            seed=1,
        )
        results = evaluator.run()
        cfg = results["config"]
        assert cfg["protocol"] == "loso"
        assert cfg["seed"] == 1
        assert cfg["model_name"] == "_TestLogReg"


# ---------------------------------------------------------------------------
# Tests: SubjectDependentEvaluator
# ---------------------------------------------------------------------------


class TestSubjectDependentEvaluator:
    """Tests for the ``SubjectDependentEvaluator``."""

    def test_full_run(self) -> None:
        ds = SyntheticDataset(n_subjects=4, n_trials_per_subject=30, seed=99)
        pipeline = FeaturePipeline([("id", IdentityTransform())])
        evaluator = SubjectDependentEvaluator(
            dataset=ds,
            feature_pipeline=pipeline,
            model_config={"n_classes": 2},
            model_name="_TestLogReg",
            seed=99,
        )
        results = evaluator.run()

        assert len(results["per_subject"]) == 4
        assert "mean" in results
        assert "std" in results


# ---------------------------------------------------------------------------
# Tests: ConfigLoader
# ---------------------------------------------------------------------------


class TestConfigLoader:
    """Tests for YAML config loading and validation."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        yaml_content = """\
experiment:
  name: test_exp
  seed: 42
  device: cpu

dataset:
  name: DEAP
  root: /tmp/data

model:
  name: DGCNN
  params:
    n_classes: 2

evaluation:
  protocol: loso
  val_fraction: 0.1

output:
  results_dir: /tmp/results
  save_checkpoints: false
"""
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml_content, encoding="utf-8")

        cfg = ConfigLoader.load(str(cfg_path))
        assert cfg.experiment.name == "test_exp"
        assert cfg.experiment.seed == 42
        assert cfg.dataset.name == "DEAP"
        assert cfg.model.name == "DGCNN"
        assert cfg.evaluation.protocol == "loso"

    def test_missing_required_field(self, tmp_path: Path) -> None:
        yaml_content = """\
experiment:
  seed: 42
dataset:
  name: DEAP
model:
  name: X
"""
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text(yaml_content, encoding="utf-8")

        with pytest.raises(EmoKitConfigError, match="validation failed"):
            ConfigLoader.load(str(cfg_path))

    def test_invalid_protocol(self, tmp_path: Path) -> None:
        yaml_content = """\
experiment:
  name: x
  seed: 1
dataset:
  name: DEAP
model:
  name: X
evaluation:
  protocol: invalid_proto
"""
        cfg_path = tmp_path / "bad_proto.yaml"
        cfg_path.write_text(yaml_content, encoding="utf-8")

        with pytest.raises(EmoKitConfigError):
            ConfigLoader.load(str(cfg_path))

    def test_invalid_overlap(self, tmp_path: Path) -> None:
        yaml_content = """\
experiment:
  name: x
  seed: 1
dataset:
  name: DEAP
  overlap: 1.5
model:
  name: X
"""
        cfg_path = tmp_path / "bad_overlap.yaml"
        cfg_path.write_text(yaml_content, encoding="utf-8")

        with pytest.raises(EmoKitConfigError):
            ConfigLoader.load(str(cfg_path))

    def test_file_not_found(self) -> None:
        with pytest.raises(EmoKitConfigError, match="not found"):
            ConfigLoader.load("/nonexistent/path.yaml")

    def test_invalid_yaml_syntax(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "broken.yaml"
        cfg_path.write_text("{{{{not yaml", encoding="utf-8")
        with pytest.raises(EmoKitConfigError, match="Invalid YAML"):
            ConfigLoader.load(str(cfg_path))

    def test_non_mapping_yaml(self, tmp_path: Path) -> None:
        cfg_path = tmp_path / "list.yaml"
        cfg_path.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(EmoKitConfigError, match="mapping"):
            ConfigLoader.load(str(cfg_path))

    def test_defaults_applied(self, tmp_path: Path) -> None:
        yaml_content = """\
experiment:
  name: minimal
dataset:
  name: DEAP
model:
  name: DGCNN
"""
        cfg_path = tmp_path / "minimal.yaml"
        cfg_path.write_text(yaml_content, encoding="utf-8")

        cfg = ConfigLoader.load(str(cfg_path))
        assert cfg.experiment.seed == 42
        assert cfg.experiment.device == "cpu"
        assert cfg.evaluation.protocol == "loso"
        assert cfg.evaluation.val_fraction == 0.1
        assert cfg.output.results_dir == "results/"
        assert cfg.output.save_checkpoints is False

    def test_feature_pipeline_config(self, tmp_path: Path) -> None:
        yaml_content = """\
experiment:
  name: pipe_test
dataset:
  name: DEAP
feature_pipeline:
  steps:
    - name: DEExtractor
      params:
        fs: 128
    - name: EEGNormalizer
      params: {}
model:
  name: DGCNN
"""
        cfg_path = tmp_path / "pipe.yaml"
        cfg_path.write_text(yaml_content, encoding="utf-8")

        cfg = ConfigLoader.load(str(cfg_path))
        assert len(cfg.feature_pipeline.steps) == 2
        assert cfg.feature_pipeline.steps[0].name == "DEExtractor"
        assert cfg.feature_pipeline.steps[0].params["fs"] == 128


# ---------------------------------------------------------------------------
# Tests: ResultLogger
# ---------------------------------------------------------------------------


class TestResultLogger:
    """Tests for result persistence."""

    def _sample_results(self) -> dict[str, Any]:
        return {
            "per_subject": {
                1: {"accuracy": 0.8, "f1_macro": 0.79, "f1_weighted": 0.80},
                2: {"accuracy": 0.7, "f1_macro": 0.69, "f1_weighted": 0.71},
            },
            "mean": {"accuracy": 0.75, "f1_macro": 0.74, "f1_weighted": 0.755},
            "std": {"accuracy": 0.05, "f1_macro": 0.05, "f1_weighted": 0.045},
            "config": {
                "dataset_name": "Synthetic",
                "model_name": "TestModel",
                "protocol": "loso",
                "seed": 42,
            },
        }

    def test_log_creates_files(self, tmp_path: Path) -> None:
        rl = ResultLogger(results_dir=str(tmp_path / "out"))
        results = self._sample_results()
        json_path = rl.log(results)

        assert json_path.exists()
        assert json_path.suffix == ".json"

        csv_files = list(Path(tmp_path / "out").glob("*.csv"))
        assert len(csv_files) >= 2  # per-subject CSV + results_db.csv

    def test_json_content_valid(self, tmp_path: Path) -> None:
        rl = ResultLogger(results_dir=str(tmp_path / "out"))
        results = self._sample_results()
        json_path = rl.log(results)

        loaded = json.loads(json_path.read_text(encoding="utf-8"))
        assert loaded["mean"]["accuracy"] == pytest.approx(0.75)
        assert "per_subject" in loaded

    def test_db_appends(self, tmp_path: Path) -> None:
        rl = ResultLogger(results_dir=str(tmp_path / "out"))
        rl.log(self._sample_results())
        rl.log(self._sample_results())

        db_path = tmp_path / "out" / "results_db.csv"
        lines = db_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3  # header + 2 rows

    def test_creates_results_dir(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "dir"
        assert not target.exists()
        _rl = ResultLogger(results_dir=str(target))
        assert target.exists()


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case tests for evaluation protocols."""

    def test_single_subject_loso(self) -> None:
        """LOSO with a single subject produces empty results (no train data)."""
        ds = SyntheticDataset(n_subjects=1, n_trials_per_subject=20, seed=0)
        pipeline = FeaturePipeline([("id", IdentityTransform())])
        evaluator = LOSOEvaluator(
            dataset=ds,
            feature_pipeline=pipeline,
            model_config={"n_classes": 2},
            model_name="_TestLogReg",
            seed=0,
        )
        results = evaluator.run()
        assert len(results["per_subject"]) == 0
        assert results["mean"] == {}
        assert results["std"] == {}

    def test_imbalanced_classes(self) -> None:
        """Verify evaluator handles heavily imbalanced labels."""
        ds = SyntheticDataset(n_subjects=3, n_trials_per_subject=40, seed=77)
        for sid in ds._data:
            X, y = ds._data[sid]
            y[:] = 0
            y[:3] = 1
            ds._data[sid] = (X, y)

        pipeline = FeaturePipeline([("id", IdentityTransform())])
        evaluator = LOSOEvaluator(
            dataset=ds,
            feature_pipeline=pipeline,
            model_config={"n_classes": 2},
            model_name="_TestLogReg",
            seed=77,
        )
        results = evaluator.run()
        assert len(results["per_subject"]) == 3
        for m in results["per_subject"].values():
            assert 0.0 <= m["accuracy"] <= 1.0

    def test_reproducibility(self) -> None:
        """Two runs with the same seed must yield identical results."""
        def _run(seed: int) -> dict[str, Any]:
            ds = SyntheticDataset(n_subjects=3, n_trials_per_subject=20, seed=seed)
            pipeline = FeaturePipeline([("id", IdentityTransform())])
            evaluator = LOSOEvaluator(
                dataset=ds,
                feature_pipeline=pipeline,
                model_config={"n_classes": 2},
                model_name="_TestLogReg",
                seed=seed,
            )
            return evaluator.run()

        r1 = _run(42)
        r2 = _run(42)
        assert r1["mean"] == r2["mean"]


# ---------------------------------------------------------------------------
# Tests: pydantic model construction
# ---------------------------------------------------------------------------


class TestPydanticModels:
    """Direct pydantic model validation tests."""

    def test_full_config_construction(self) -> None:
        cfg = FullConfig(
            experiment=ExperimentConfig(name="t"),
            dataset=DatasetConfig(name="D"),
            model=ModelConfig(name="M"),
        )
        assert cfg.evaluation.protocol == "loso"
        assert cfg.output.results_dir == "results/"

    def test_invalid_val_fraction(self) -> None:
        with pytest.raises(Exception):
            EvaluationConfig(protocol="loso", val_fraction=1.5)

    def test_feature_step_default_params(self) -> None:
        step = FeatureStepConfig(name="Foo")
        assert step.params == {}


# ---------------------------------------------------------------------------
# Paper-aligned evaluation tests (P0-6)
# ---------------------------------------------------------------------------


class TestLOSOPerSubjectRawPreds:
    """Verify per_subject_raw_preds is in LOSO results."""

    def test_raw_preds_in_results(self) -> None:
        ds = SyntheticDataset(n_subjects=3, n_trials_per_subject=20, seed=42)
        pipeline = FeaturePipeline([("identity", IdentityTransform())])
        evaluator = LOSOEvaluator(
            dataset=ds,
            feature_pipeline=pipeline,
            model_config={"n_classes": 2},
            model_name="_TestLogReg",
            seed=42,
        )
        results = evaluator.run()
        assert "per_subject_raw_preds" in results
        for sid in results["per_subject"]:
            assert sid in results["per_subject_raw_preds"]
            raw = results["per_subject_raw_preds"][sid]
            assert "y_true" in raw
            assert "y_pred" in raw
            assert len(raw["y_true"]) == len(raw["y_pred"])


class TestWilcoxonScriptRuns:
    """Script must run without error on synthetic input."""

    def test_wilcoxon_script_runs(self, tmp_path: Path) -> None:
        from emokit.scripts.statistical_analysis import run_pairwise_wilcoxon

        mock = {
            m: {str(s): {"accuracy": float(np.random.rand())} for s in range(10)}
            for m in ["CNN-LSTM", "DGCNN", "PR-PL"]
        }
        p = tmp_path / "mock.json"
        p.write_text(json.dumps(mock), encoding="utf-8")
        results = run_pairwise_wilcoxon(str(p), alpha=0.05)
        assert "CNN-LSTM vs DGCNN" in results
        assert "p" in results["CNN-LSTM vs DGCNN"]
