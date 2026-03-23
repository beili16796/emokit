# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""End-to-end integration test for the EmoKit pipeline.

Generates a synthetic DEAP-like dataset, runs DEExtractor → EEGNormalizer
feature extraction through the LOSOEvaluator with a CNN-LSTM model, and
validates the result dict structure.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from emokit.datasets.base import BaseDataset, _REGISTRY
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


# ---------------------------------------------------------------------------
# Adapter transforms -- LOSOEvaluator flattens 3-D raw data to 2-D before
# the feature pipeline, so we restore the channel/time axes for DEExtractor
# and flatten back to 2-D for the DE-input CNN-LSTM afterwards.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------


@_REGISTRY.register("MockDEAP")
class MockDEAPDataset(BaseDataset):
    """Synthetic DEAP-like dataset for integration testing.

    Generates random EEG data for *N_SUBJECTS* subjects, each with
    *N_TRIALS* trials of shape ``(N_TRIALS, N_CHANNELS, N_SAMPLES)``
    and binary labels.
    """

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


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


class TestIntegration:
    """Full pipeline integration: synthetic data → DE features → CNN-LSTM → LOSO."""

    def test_loso_cnn_lstm_pipeline(self) -> None:
        set_seed(SEED)

        ds = MockDEAPDataset()

        assert ds.get_subject_ids() == [1, 2, 3]
        raw = ds.read_raw(1)
        assert raw["eeg"].shape == (N_TRIALS, N_CHANNELS, N_SAMPLES)
        assert raw["labels"].shape == (N_TRIALS,)

        pipeline = FeaturePipeline([
            ("reshape", _ReshapeTo3D(N_CHANNELS, N_SAMPLES)),
            ("de", DEExtractor(fs=FS)),
            ("norm", EEGNormalizer()),
            ("flatten", _FlattenTo2D()),
        ])

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

        assert len(results["per_subject"]) == N_SUBJECTS
        for sid in range(1, N_SUBJECTS + 1):
            assert sid in results["per_subject"]
            metrics = results["per_subject"][sid]
            assert "accuracy" in metrics
            assert "f1_macro" in metrics
            assert "f1_weighted" in metrics
            assert 0.0 <= metrics["accuracy"] <= 1.0

        assert 0.0 <= results["mean"]["accuracy"] <= 1.0
        assert results["std"]["accuracy"] >= 0.0

        cfg = results["config"]
        assert cfg["protocol"] == "loso"
        assert cfg["model_name"] == "CNN-LSTM"
        assert cfg["seed"] == SEED
