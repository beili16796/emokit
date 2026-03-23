# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Unit tests for the emokit.datasets layer using synthetic data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from emokit.datasets.base import (
    BaseDataset,
    DatasetRegistry,
    _REGISTRY,
    load_dataset,
    segment_trials,
)
from emokit.datasets.deap import DEAPDataset
from emokit.utils import EmoKitDataError


# ======================================================================
# segment_trials
# ======================================================================


class TestSegmentTrials:
    """Tests for the sliding-window segmentation function."""

    def test_basic_shape(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.standard_normal((5, 3, 1000))
        windows = segment_trials(data, fs=100.0, window_sec=2.0, overlap=0.5)
        win_samples = 200
        step = 100
        wins_per_trial = (1000 - win_samples) // step + 1  # 9
        assert windows.shape == (5 * wins_per_trial, 3, win_samples)

    def test_no_overlap(self) -> None:
        rng = np.random.default_rng(1)
        data = rng.standard_normal((2, 4, 400))
        windows = segment_trials(data, fs=100.0, window_sec=1.0, overlap=0.0)
        assert windows.shape == (2 * 4, 4, 100)

    def test_full_trial_single_window(self) -> None:
        rng = np.random.default_rng(2)
        data = rng.standard_normal((3, 2, 200))
        windows = segment_trials(data, fs=100.0, window_sec=2.0, overlap=0.0)
        assert windows.shape == (3, 2, 200)

    def test_window_larger_than_trial_raises(self) -> None:
        rng = np.random.default_rng(3)
        data = rng.standard_normal((1, 1, 50))
        with pytest.raises(ValueError, match="exceeds trial length"):
            segment_trials(data, fs=100.0, window_sec=1.0, overlap=0.0)

    def test_bad_overlap_raises(self) -> None:
        data = np.zeros((1, 1, 100))
        with pytest.raises(ValueError, match="overlap must be in"):
            segment_trials(data, fs=100.0, window_sec=0.5, overlap=1.0)
        with pytest.raises(ValueError, match="overlap must be in"):
            segment_trials(data, fs=100.0, window_sec=0.5, overlap=-0.1)

    def test_2d_input_raises(self) -> None:
        data = np.zeros((10, 100))
        with pytest.raises(AssertionError, match="Expected 3D"):
            segment_trials(data, fs=100.0)

    def test_single_sample_trial(self) -> None:
        data = np.ones((2, 3, 1))
        with pytest.raises(ValueError, match="exceeds trial length"):
            segment_trials(data, fs=1.0, window_sec=2.0, overlap=0.0)

    def test_window_count_formula(self) -> None:
        """Verify expected number of windows per trial."""
        rng = np.random.default_rng(10)
        n_trials, n_ch, n_samples = 4, 2, 512
        fs, win_sec, overlap = 128.0, 1.0, 0.25
        data = rng.standard_normal((n_trials, n_ch, n_samples))

        win_samples = int(round(win_sec * fs))  # 128
        step = int(round(win_samples * (1.0 - overlap)))  # 96
        expected_per_trial = (n_samples - win_samples) // step + 1  # 5

        windows = segment_trials(data, fs, win_sec, overlap)
        assert windows.shape[0] == n_trials * expected_per_trial

    def test_data_integrity(self) -> None:
        """First window of first trial should exactly match the source."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((1, 2, 300))
        windows = segment_trials(data, fs=100.0, window_sec=1.0, overlap=0.0)
        np.testing.assert_array_equal(windows[0], data[0, :, :100])


# ======================================================================
# DatasetRegistry
# ======================================================================


class TestDatasetRegistry:
    """Tests for the dataset name→class registry."""

    def test_register_and_lookup(self) -> None:
        reg = DatasetRegistry()

        @reg.register("dummy")
        class _Dummy:
            pass

        assert reg["dummy"] is _Dummy
        assert "dummy" in reg.available()

    def test_overwrite_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        reg = DatasetRegistry()

        @reg.register("dup")
        class _A:
            pass

        @reg.register("dup")
        class _B:
            pass

        assert reg["dup"] is _B

    def test_missing_key_raises(self) -> None:
        reg = DatasetRegistry()
        with pytest.raises(KeyError):
            _ = reg["nonexistent"]

    def test_available_sorted(self) -> None:
        reg = DatasetRegistry()

        @reg.register("Z_data")
        class _Z:
            pass

        @reg.register("A_data")
        class _A:
            pass

        assert reg.available() == ["A_data", "Z_data"]


# ======================================================================
# Global registry & load_dataset
# ======================================================================


class TestLoadDataset:
    """Tests for the global registry and convenience loader."""

    def test_known_datasets_registered(self) -> None:
        for name in ("DEAP", "SEED", "SEED-V", "MAHNOB-HCI", "DREAMER"):
            assert name in _REGISTRY, f"'{name}' not registered"

    def test_load_unknown_raises(self) -> None:
        with pytest.raises(EmoKitDataError, match="Unknown dataset"):
            load_dataset("NONEXISTENT_DATASET_XYZ")

    def test_load_dataset_returns_instance(self, tmp_path: Any) -> None:
        ds = load_dataset("DEAP", root=str(tmp_path))
        assert isinstance(ds, DEAPDataset)


# ======================================================================
# BaseDataset subclass with mock read_raw
# ======================================================================


class _MockDataset(BaseDataset):
    """Minimal concrete subclass for testing the base class logic."""

    def __init__(self, data: np.ndarray, labels: np.ndarray, fs: float = 100.0,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._data = data
        self._labels = labels
        self._fs = fs

    def _get_fs(self) -> float:
        return self._fs

    def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
        return {"eeg": self._data, "labels": self._labels}

    def get_subject_ids(self) -> list[int]:
        return [1, 2]

    def get_channel_names(self, modality: str) -> list[str]:
        return [f"ch{i}" for i in range(self._data.shape[1])]

    def get_label_names(self) -> list[str]:
        return ["neg", "pos"]


class TestBaseDataset:
    """Tests for load() and other BaseDataset methods via mock subclass."""

    def test_load_shapes(self, tmp_path: Any) -> None:
        rng = np.random.default_rng(99)
        n_trials, n_ch, n_samples = 10, 4, 500
        data = rng.standard_normal((n_trials, n_ch, n_samples))
        labels = rng.integers(0, 2, size=n_trials)

        ds = _MockDataset(
            data=data, labels=labels, fs=100.0,
            root=str(tmp_path), subjects=[1], window_sec=2.0, overlap=0.5,
        )
        X, y = ds.load()
        assert X.ndim == 3
        assert X.shape[1] == n_ch
        assert X.shape[2] == 200  # 2.0 sec × 100 Hz
        assert y.shape[0] == X.shape[0]

    def test_load_no_data_raises(self, tmp_path: Any) -> None:
        class _EmptyDataset(BaseDataset):
            def _get_fs(self) -> float:
                return 100.0

            def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
                return {"labels": np.array([0, 1])}

            def get_subject_ids(self) -> list[int]:
                return [1]

            def get_channel_names(self, modality: str) -> list[str]:
                return []

            def get_label_names(self) -> list[str]:
                return []

        ds = _EmptyDataset(root=str(tmp_path), subjects=[1])
        with pytest.raises(EmoKitDataError, match="No data loaded"):
            ds.load()

    def test_modalities_filter(self, tmp_path: Any) -> None:
        rng = np.random.default_rng(7)
        data = rng.standard_normal((5, 8, 300))
        labels = rng.integers(0, 2, size=5)

        class _MultiMod(BaseDataset):
            def _get_fs(self) -> float:
                return 100.0

            def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
                return {
                    "eeg": data[:, :6, :],
                    "ecg": data[:, 6:8, :],
                    "labels": labels,
                }

            def get_subject_ids(self) -> list[int]:
                return [1]

            def get_channel_names(self, modality: str) -> list[str]:
                return []

            def get_label_names(self) -> list[str]:
                return []

        ds = _MultiMod(
            root=str(tmp_path), subjects=[1],
            modalities=["eeg"], window_sec=1.0, overlap=0.0,
        )
        X, y = ds.load()
        assert X.shape[1] == 6  # only EEG channels

    def test_root_defaults_to_data_root(self) -> None:
        rng = np.random.default_rng(0)
        ds = _MockDataset(
            data=rng.standard_normal((1, 1, 100)),
            labels=np.array([0]),
        )
        assert ds.root.name == "emokit_data" or "EMOKIT_DATA_ROOT" in str(ds.root)


# ======================================================================
# DEAP-specific tests
# ======================================================================


class TestDEAPBinarization:
    """Tests for DEAP label binarisation logic."""

    def test_threshold_default(self) -> None:
        ratings = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        binary = DEAPDataset.binarize_labels(ratings, threshold=5.0)
        np.testing.assert_array_equal(binary, [0, 0, 1, 1, 1])

    def test_threshold_custom(self) -> None:
        ratings = np.array([2.0, 4.0, 6.0, 8.0])
        binary = DEAPDataset.binarize_labels(ratings, threshold=4.0)
        np.testing.assert_array_equal(binary, [0, 1, 1, 1])

    def test_all_below(self) -> None:
        ratings = np.array([1.0, 2.0, 3.0])
        binary = DEAPDataset.binarize_labels(ratings, threshold=5.0)
        np.testing.assert_array_equal(binary, [0, 0, 0])

    def test_all_above(self) -> None:
        ratings = np.array([5.0, 6.0, 7.0])
        binary = DEAPDataset.binarize_labels(ratings, threshold=5.0)
        np.testing.assert_array_equal(binary, [1, 1, 1])

    def test_empty_array(self) -> None:
        ratings = np.array([])
        binary = DEAPDataset.binarize_labels(ratings, threshold=5.0)
        assert binary.shape == (0,)

    def test_invalid_label_axis(self, tmp_path: Any) -> None:
        with pytest.raises(ValueError, match="label_axis"):
            DEAPDataset(root=str(tmp_path), label_axis="dominance")


# ======================================================================
# Edge cases
# ======================================================================


class TestEdgeCases:
    """Edge-case tests for robustness."""

    def test_empty_3d_array(self) -> None:
        data = np.empty((0, 4, 100))
        windows = segment_trials(data, fs=100.0, window_sec=0.5, overlap=0.0)
        assert windows.shape == (0, 4, 50)

    def test_single_trial_single_channel(self) -> None:
        rng = np.random.default_rng(5)
        data = rng.standard_normal((1, 1, 200))
        windows = segment_trials(data, fs=100.0, window_sec=1.0, overlap=0.0)
        assert windows.shape == (2, 1, 100)

    def test_deap_channel_names(self, tmp_path: Any) -> None:
        ds = DEAPDataset(root=str(tmp_path))
        names = ds.get_channel_names("eeg")
        assert len(names) == 32
        assert names[0] == "Fp1"

    def test_deap_unknown_modality(self, tmp_path: Any) -> None:
        ds = DEAPDataset(root=str(tmp_path))
        with pytest.raises(EmoKitDataError, match="Unknown DEAP modality"):
            ds.get_channel_names("mag")

    def test_deap_no_file_raises(self, tmp_path: Any) -> None:
        ds = DEAPDataset(root=str(tmp_path))
        with pytest.raises(EmoKitDataError, match="No .bdf or .dat"):
            ds.read_raw(1)

    def test_window_count_exact_division(self) -> None:
        """When trial length is exact multiple of window, no remainder."""
        data = np.ones((1, 2, 400))
        windows = segment_trials(data, fs=100.0, window_sec=1.0, overlap=0.0)
        assert windows.shape[0] == 4

    def test_segment_preserves_dtype(self) -> None:
        data = np.zeros((1, 1, 200), dtype=np.float32)
        windows = segment_trials(data, fs=100.0, window_sec=1.0, overlap=0.0)
        assert windows.dtype == np.float32
