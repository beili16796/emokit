# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Tests for CrossCorpusEvaluator and data augmentation modules."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from emokit.datasets.base import BaseDataset
from emokit.evaluation.protocols import CrossCorpusEvaluator
from emokit.features.augmentation import (
    FeatureMixup,
    TemporalSegmentPermutation,
)
from emokit.features.base import FeaturePipeline
from emokit.features.channel_align import align_channels, subset_features
from emokit.utils import set_seed

SEED = 42


# ── Channel alignment ────────────────────────────────────────────────


class TestChannelAlign:
    def test_exact_match(self) -> None:
        names = ["Fp1", "Fp2", "F3", "F4"]
        idx = align_channels(names, names)
        assert idx == [0, 1, 2, 3]

    def test_subset(self) -> None:
        source = ["Fp1", "AF3", "F3", "F7", "FC5", "T7", "P7", "O1"]
        target = ["F3", "O1"]
        idx = align_channels(source, target)
        assert idx == [2, 7]

    def test_missing_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            align_channels(["Fp1", "Fp2"], ["Cz"])

    def test_subset_features_shape(self) -> None:
        X = np.random.randn(10, 8, 5)
        src = ["Fp1", "AF3", "F3", "F7", "FC5", "T7", "P7", "O1"]
        tgt = ["F3", "O1"]
        out = subset_features(X, src, tgt)
        assert out.shape == (10, 2, 5)
        np.testing.assert_array_equal(out[:, 0, :], X[:, 2, :])
        np.testing.assert_array_equal(out[:, 1, :], X[:, 7, :])


# ── Augmentation ──────────────────────────────────────────────────────


class TestFeatureMixup:
    def test_output_larger(self) -> None:
        X = np.random.randn(20, 10)
        aug = FeatureMixup(alpha=0.2, ratio=0.5, seed=SEED)
        X_out = aug.fit_transform(X)
        assert X_out.shape[0] == 20 + 10
        assert X_out.shape[1] == 10

    def test_inference_noop(self) -> None:
        X = np.random.randn(20, 10)
        aug = FeatureMixup(alpha=0.2, ratio=0.5, seed=SEED)
        aug.fit_transform(X)
        X_test = np.random.randn(5, 10)
        X_out = aug.transform(X_test)
        np.testing.assert_array_equal(X_out, X_test)

    def test_no_nan(self) -> None:
        X = np.random.randn(50, 32, 5)
        out = FeatureMixup(alpha=0.4, ratio=1.0).fit_transform(X)
        assert not np.any(np.isnan(out))


class TestTemporalSegmentPermutation:
    def test_output_larger(self) -> None:
        X = np.random.randn(10, 14, 512)
        aug = TemporalSegmentPermutation(n_segments=4, ratio=0.5, seed=SEED)
        X_out = aug.fit_transform(X)
        assert X_out.shape[0] == 10 + 5
        assert X_out.shape[1:] == (14, 512)

    def test_inference_noop(self) -> None:
        X = np.random.randn(10, 14, 512)
        aug = TemporalSegmentPermutation(n_segments=4, ratio=0.5)
        aug.fit_transform(X)
        X_test = np.random.randn(3, 14, 512)
        np.testing.assert_array_equal(aug.transform(X_test), X_test)

    def test_2d_passthrough(self) -> None:
        X = np.random.randn(10, 50)
        out = TemporalSegmentPermutation().fit_transform(X)
        np.testing.assert_array_equal(out, X)


# ── CrossCorpusEvaluator ─────────────────────────────────────────────


class _MockDataset(BaseDataset):
    """Tiny dataset for cross-corpus testing."""

    def __init__(
        self,
        n_subjects: int = 3,
        n_trials: int = 10,
        n_channels: int = 32,
        channel_names: list[str] | None = None,
        n_classes: int = 2,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(root="/tmp/mock_cc", **kwargs)
        self._n_subjects = n_subjects
        self._n_trials = n_trials
        self._n_channels = n_channels
        self._ch_names = channel_names or [f"ch{i}" for i in range(n_channels)]
        self._n_classes = n_classes
        self._seed = seed

    def _get_fs(self) -> float:
        return 128.0

    def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
        rng = np.random.RandomState(self._seed + subject_id)
        X = rng.randn(self._n_trials, 1, self._n_channels).astype(np.float32)
        y = rng.randint(0, self._n_classes, size=self._n_trials)
        y[0] = 0
        y[1] = min(1, self._n_classes - 1)
        return {"eeg": X, "labels": y}

    def get_subject_ids(self) -> list[int]:
        return list(range(1, self._n_subjects + 1))

    def get_channel_names(self, modality: str) -> list[str]:
        return list(self._ch_names)

    def get_label_names(self) -> list[str]:
        return [f"c{i}" for i in range(self._n_classes)]


class _Identity:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def fit_transform(self, X, y=None):
        return X


class TestCrossCorpusEvaluator:
    def test_same_montage(self) -> None:
        set_seed(SEED)
        src = _MockDataset(n_subjects=3, n_trials=12, n_channels=10)
        tgt = _MockDataset(n_subjects=2, n_trials=8, n_channels=10, seed=99)
        pipeline = FeaturePipeline([("id", _Identity())])
        ev = CrossCorpusEvaluator(
            source_dataset=src,
            target_dataset=tgt,
            feature_pipeline=pipeline,
            model_config={
                "n_classes": 2,
                "n_feat": 10,
                "n_epochs": 1,
                "batch_size": 8,
            },
            model_name="PR-PL",
            seed=SEED,
            val_fraction=0.0,
        )
        results = ev.run()
        assert results["config"]["protocol"] == "cross_corpus"
        assert len(results["per_subject"]) == 2
        assert 0.0 <= results["mean"]["accuracy"] <= 1.0

    def test_channel_alignment(self) -> None:
        set_seed(SEED)
        shared = [
            "Fp1",
            "AF3",
            "F3",
            "FC5",
            "T7",
            "P7",
            "O1",
            "O2",
            "P8",
            "T8",
            "FC6",
            "F4",
            "F8",
            "AF4",
        ]
        src_ch = shared + [
            "F7",
            "FC1",
            "C3",
            "CP5",
            "CP1",
            "P3",
            "PO3",
            "Oz",
            "Pz",
            "Fp2",
            "F8",
            "FC2",
            "C4",
            "T8_dup",
            "CP6",
            "CP2",
            "P4",
            "P8_dup",
        ]
        src = _MockDataset(
            n_subjects=3,
            n_trials=12,
            n_channels=len(src_ch),
            channel_names=src_ch,
        )
        tgt = _MockDataset(
            n_subjects=2,
            n_trials=8,
            n_channels=14,
            channel_names=shared,
            seed=77,
        )
        pipeline = FeaturePipeline([("id", _Identity())])
        ev = CrossCorpusEvaluator(
            source_dataset=src,
            target_dataset=tgt,
            feature_pipeline=pipeline,
            model_config={
                "n_classes": 2,
                "n_feat": 14,
                "n_epochs": 1,
                "batch_size": 8,
            },
            model_name="PR-PL",
            seed=SEED,
            val_fraction=0.0,
        )
        results = ev.run()
        assert results["config"]["channel_alignment"] is True
        assert results["config"]["source_channels"] == len(src_ch)
        assert results["config"]["target_channels"] == 14
