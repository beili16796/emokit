# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""End-to-end integration tests for the EmoKit pipeline."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from emokit.datasets.base import _REGISTRY, BaseDataset
from emokit.evaluation.protocols import LOSOEvaluator
from emokit.features.base import BaseTransform, FeaturePipeline
from emokit.features.eeg import DEExtractor, EEGNormalizer
from emokit.utils import set_seed

N_SUBJECTS = 3
N_TRIALS = 5
N_CHANNELS = 32
FS = 128
WINDOW_SEC = 4.0
N_SAMPLES = int(FS * WINDOW_SEC)  # 512
N_CLASSES = 2
SEED = 42


class _ReshapeTo3D(BaseTransform):
    """Reshape flattened ``(N, C*T)`` back to ``(N, C, T)``."""

    def __init__(self, n_channels: int, n_samples: int) -> None:
        self.n_channels = n_channels
        self.n_samples = n_samples

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> _ReshapeTo3D:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            return X.reshape(X.shape[0], self.n_channels, self.n_samples)
        return X


class _FlattenTo2D(BaseTransform):
    """Flatten ``(N, C, F)`` to ``(N, C*F)``."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> _FlattenTo2D:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X


if "MockDEAP" not in _REGISTRY:

    @_REGISTRY.register("MockDEAP")
    class MockDEAPDataset(BaseDataset):
        """Synthetic DEAP-like dataset for integration testing."""

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(root="/tmp/mock_deap", **kwargs)
            rng = np.random.RandomState(SEED)
            self._data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for sid in range(1, N_SUBJECTS + 1):
                eeg = rng.randn(N_TRIALS, N_CHANNELS, N_SAMPLES).astype(np.float32)
                labels = rng.randint(0, N_CLASSES, size=N_TRIALS).astype(np.int64)
                labels[0] = 0
                labels[1] = 1
                self._data[sid] = (eeg, labels)

        def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
            eeg, labels = self._data[subject_id]
            return {"eeg": eeg, "labels": labels}

        def get_subject_ids(self) -> list[int]:
            return list(range(1, N_SUBJECTS + 1))

        def get_channel_names(self, modality: str) -> list[str]:
            return [f"ch{i}" for i in range(N_CHANNELS)]

        def get_label_names(self) -> list[str]:
            return ["low", "high"]

        def _get_fs(self) -> float:
            return float(FS)

else:
    MockDEAPDataset = _REGISTRY["MockDEAP"]


class TestIntegration:
    """Full pipeline integration: synthetic data -> DE features -> CNN-LSTM -> LOSO."""

    def test_loso_cnn_lstm_pipeline(self) -> None:
        set_seed(SEED)
        ds = MockDEAPDataset()

        pipeline = FeaturePipeline(
            [
                ("de", DEExtractor(fs=FS)),
                ("norm", EEGNormalizer()),
                ("flatten", _FlattenTo2D()),
            ]
        )

        model_config: dict[str, Any] = {
            "n_classes": N_CLASSES,
            "input_type": "de",
            "n_channels": N_CHANNELS,
            "n_epochs": 1,
            "batch_size": 4,
            "hidden_size": 16,
            "n_layers": 1,
            "dropout": 0.0,
            "lr": 1e-3,
            "device": "cpu",
        }

        evaluator = LOSOEvaluator(
            dataset=ds,
            feature_pipeline=pipeline,
            model_config=model_config,
            model_name="CNN-LSTM",
            seed=SEED,
            val_fraction=0.0,
        )

        results = evaluator.run()

        assert "per_subject" in results
        assert "mean" in results
        assert "std" in results
        assert "config" in results
        assert "per_subject_raw_preds" in results

        assert len(results["per_subject"]) == N_SUBJECTS
        for sid in range(1, N_SUBJECTS + 1):
            assert sid in results["per_subject"]
            metrics = results["per_subject"][sid]
            assert "accuracy" in metrics
            assert "f1_macro" in metrics
            assert "f1_weighted" in metrics
            assert 0.0 <= metrics["accuracy"] <= 1.0

        assert 0.0 <= results["mean"]["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# Paper-aligned integration tests (P4-1)
# ---------------------------------------------------------------------------


def test_quick_demo_end_to_end():
    """The README quickstart must work exactly as written."""
    result = subprocess.run(
        [sys.executable, "-m", "emokit.run", "configs/quick_demo.yaml", "--dry-run"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"Exit {result.returncode}:\n{result.stderr}"
    out = result.stdout.lower() + result.stderr.lower()
    assert "accuracy" in out


def test_all_models_predict_on_synthetic():
    """Every model in ModelRegistry must produce valid predictions."""
    from emokit.datasets.synthetic import SyntheticDataset
    from emokit.models.base import build_model

    ds = SyntheticDataset(n_subjects=3, n_trials=10, n_classes=2, seed=0)
    de = DEExtractor(fs=128)

    for name in ["CNN-LSTM", "DGCNN", "Transformer-MM", "BiDAE", "DGCCA-AM", "PR-PL"]:
        model_kwargs: dict[str, Any] = {
            "n_classes": 2,
            "n_epochs": 1,
            "batch_size": 8,
        }

        if name == "CNN-LSTM":
            model_kwargs.update({"input_type": "de", "n_channels": 32})
        elif name == "DGCNN":
            model_kwargs.update({"n_channels": 32, "n_bands": 5})
        elif name == "Transformer-MM":
            model_kwargs.update(
                {
                    "n_channels": 32,
                    "n_bands": 5,
                    "n_peripheral_feat": 7,
                }
            )
        elif name == "BiDAE":
            model_kwargs.update({"n_feat_mod1": 160, "n_feat_mod2": 7})
        elif name == "DGCCA-AM":
            model_kwargs.update(
                {
                    "n_feat_eeg": 160,
                    "n_feat_gsr": 32,
                    "n_feat_ecg": 16,
                }
            )
        elif name == "PR-PL":
            model_kwargs.update({"n_feat": 160, "prototype_dim": 64})

        model = build_model(name, model_kwargs)

        X_all, y_all = [], []
        for sid in ds.get_subject_ids():
            raw = ds.read_raw(sid)
            X_subj = de.transform(raw["eeg"])
            X_all.append(X_subj)
            y_all.append(raw["labels"])
        X = np.concatenate(X_all, axis=0)
        y = np.concatenate(y_all, axis=0)

        if name in ("CNN-LSTM", "PR-PL"):
            X_flat = X.reshape(X.shape[0], -1)
            model.fit(X_flat[:20], y[:20])
            preds = model.predict(X_flat[20:])
        elif name == "DGCNN":
            model.fit(X[:20], y[:20])
            preds = model.predict(X[20:])
        elif name == "Transformer-MM":
            n_test = X[20:].shape[0]
            data = {
                "eeg": X[:20],
                "peripheral": np.random.randn(20, 7).astype(np.float32),
            }
            model.fit(data, y[:20])
            data_test = {
                "eeg": X[20:],
                "peripheral": np.random.randn(n_test, 7).astype(np.float32),
            }
            preds = model.predict(data_test)
        elif name == "BiDAE":
            data = {
                "mod1": X[:20].reshape(20, -1),
                "mod2": np.random.randn(20, 7).astype(np.float32),
            }
            model.fit(data, y[:20])
            n_test = X[20:].shape[0]
            data_test = {
                "mod1": X[20:].reshape(n_test, -1),
                "mod2": np.random.randn(n_test, 7).astype(np.float32),
            }
            preds = model.predict(data_test)
        elif name == "DGCCA-AM":
            n_train = 20
            data = {
                "eeg": X[:n_train].reshape(n_train, -1),
                "gsr": np.random.randn(n_train, 32).astype(np.float32),
                "ecg": np.random.randn(n_train, 16).astype(np.float32),
            }
            model.fit(data, y[:n_train])
            n_test = X[20:].shape[0]
            data_test = {
                "eeg": X[20:].reshape(n_test, -1),
                "gsr": np.random.randn(n_test, 32).astype(np.float32),
                "ecg": np.random.randn(n_test, 16).astype(np.float32),
            }
            preds = model.predict(data_test)

        assert preds.shape == (X[20:].shape[0],), f"{name} predict shape wrong"
        assert set(preds).issubset(set(range(2))), f"{name} invalid labels"


def test_yaml_config_validation_rejects_bad_config(tmp_path):
    """Pydantic config validation must catch bad field types."""
    from emokit.evaluation.config import ConfigLoader

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "experiment:\n  name: x\ndataset:\n  name: DEAP\n"
        "  window_sec: 'not_a_number'\nmodel:\n  name: X\n"
    )
    with pytest.raises(Exception):
        ConfigLoader.load(str(bad_yaml))
