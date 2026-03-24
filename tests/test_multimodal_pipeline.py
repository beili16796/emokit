# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Integration tests for the multimodal pipeline (BiDAE + DGCCA-AM).

Verifies that LOSOEvaluator correctly passes dict-valued inputs to
multimodal models and that feature pipelines apply per-modality transforms.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from emokit.datasets.base import BaseDataset
from emokit.evaluation.protocols import LOSOEvaluator
from emokit.features.base import BaseTransform, FeaturePipeline
from emokit.features.peripheral import ModalityFusionTransform
from emokit.utils import set_seed

N_SUBJECTS = 3
N_TRIALS = 8
N_EEG_FEAT = 50
N_GSR_FEAT = 3
N_ECG_FEAT = 5
N_CLASSES = 2
SEED = 42


class _MultimodalDataset(BaseDataset):
    """Synthetic dataset returning EEG, GSR, and ECG modalities."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(root="/tmp/mm_test", **kwargs)
        rng = np.random.RandomState(SEED)
        self._data: dict[int, dict[str, np.ndarray]] = {}
        for sid in range(1, N_SUBJECTS + 1):
            eeg = rng.randn(N_TRIALS, 1, N_EEG_FEAT).astype(np.float32)
            gsr = rng.randn(N_TRIALS, 1, N_GSR_FEAT).astype(np.float32)
            ecg = rng.randn(N_TRIALS, 1, N_ECG_FEAT).astype(np.float32)
            labels = rng.randint(0, N_CLASSES, size=N_TRIALS).astype(np.int64)
            labels[0] = 0
            labels[1] = 1
            self._data[sid] = {
                "eeg": eeg,
                "gsr": gsr,
                "ecg": ecg,
                "labels": labels,
            }

    def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
        return dict(self._data[subject_id])

    def get_subject_ids(self) -> list[int]:
        return list(range(1, N_SUBJECTS + 1))

    def get_channel_names(self, modality: str) -> list[str]:
        sizes = {"eeg": N_EEG_FEAT, "gsr": N_GSR_FEAT, "ecg": N_ECG_FEAT}
        return [f"{modality}{i}" for i in range(sizes.get(modality, 0))]

    def get_label_names(self) -> list[str]:
        return ["low", "high"]

    def _get_fs(self) -> float:
        return 128.0


class _IdentityTransform(BaseTransform):
    """Pass-through transform for testing."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> _IdentityTransform:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


class TestBiDAEPipeline:
    """Test that BiDAE receives dict input via LOSOEvaluator."""

    def test_loso_bidae(self) -> None:
        set_seed(SEED)
        ds = _MultimodalDataset(modalities=["eeg", "gsr"])
        pipeline = FeaturePipeline([("identity", _IdentityTransform())])

        model_config: dict[str, Any] = {
            "n_classes": N_CLASSES,
            "n_feat1": N_EEG_FEAT,
            "n_feat2": N_GSR_FEAT,
            "bottleneck_dim": 16,
            "n_epochs": 2,
            "batch_size": 4,
            "lr": 1e-3,
            "device": "cpu",
        }

        evaluator = LOSOEvaluator(
            dataset=ds,
            feature_pipeline=pipeline,
            model_config=model_config,
            model_name="BiDAE",
            seed=SEED,
            val_fraction=0.0,
        )
        results = evaluator.run()

        assert "per_subject" in results
        assert len(results["per_subject"]) == N_SUBJECTS
        for sid in range(1, N_SUBJECTS + 1):
            assert sid in results["per_subject"]
            assert 0.0 <= results["per_subject"][sid]["accuracy"] <= 1.0

        assert 0.0 <= results["mean"]["accuracy"] <= 1.0


class TestDGCCAAMPipeline:
    """Test that DGCCA-AM receives dict input via LOSOEvaluator."""

    def test_loso_dgccaam(self) -> None:
        set_seed(SEED)
        ds = _MultimodalDataset(modalities=["eeg", "gsr", "ecg"])
        pipeline = FeaturePipeline([("identity", _IdentityTransform())])

        model_config: dict[str, Any] = {
            "n_classes": N_CLASSES,
            "n_feat_eeg": N_EEG_FEAT,
            "n_feat_gsr": N_GSR_FEAT,
            "n_feat_ecg": N_ECG_FEAT,
            "hidden_dim": 16,
            "n_epochs": 2,
            "batch_size": 4,
            "lr": 1e-3,
            "device": "cpu",
        }

        evaluator = LOSOEvaluator(
            dataset=ds,
            feature_pipeline=pipeline,
            model_config=model_config,
            model_name="DGCCA-AM",
            seed=SEED,
            val_fraction=0.0,
        )
        results = evaluator.run()

        assert "per_subject" in results
        assert len(results["per_subject"]) == N_SUBJECTS
        assert 0.0 <= results["mean"]["accuracy"] <= 1.0

    def test_attention_weights_shape(self) -> None:
        """Verify attention weights have shape (batch, 3) and sum to 1."""
        from emokit.models.dgcca_am import DGCCAAMModel

        model = DGCCAAMModel(
            n_classes=N_CLASSES,
            n_feat_eeg=N_EEG_FEAT,
            n_feat_gsr=N_GSR_FEAT,
            n_feat_ecg=N_ECG_FEAT,
            n_epochs=1,
        )
        X = {
            "eeg": np.random.randn(8, N_EEG_FEAT).astype(np.float32),
            "gsr": np.random.randn(8, N_GSR_FEAT).astype(np.float32),
            "ecg": np.random.randn(8, N_ECG_FEAT).astype(np.float32),
        }
        weights = model.get_attention_weights(X)
        assert weights.shape == (8, 3)
        np.testing.assert_allclose(weights.sum(axis=1), np.ones(8), atol=1e-5)


class TestMultimodalFeaturePipeline:
    """Test FeaturePipeline with dict input and ModalityFusionTransform."""

    def test_dict_passthrough(self) -> None:
        """Identity transform applied per-modality preserves dict structure."""
        pipeline = FeaturePipeline([("identity", _IdentityTransform())])
        X = {
            "eeg": np.ones((4, 10)),
            "gsr": np.ones((4, 3)),
        }
        out = pipeline.fit_transform(X)
        assert isinstance(out, dict)
        assert "eeg" in out and "gsr" in out

    def test_fusion_transform(self) -> None:
        """ModalityFusionTransform concatenates modalities into flat array."""
        pipeline = FeaturePipeline([
            ("identity", _IdentityTransform()),
            ("fuse", ModalityFusionTransform()),
        ])
        X = {
            "eeg": np.ones((4, 10)),
            "gsr": np.ones((4, 3)) * 2,
        }
        out = pipeline.fit_transform(X)
        assert isinstance(out, np.ndarray)
        assert out.shape == (4, 13)
