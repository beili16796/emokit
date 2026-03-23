# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Unit tests for the feature extraction layer."""

from __future__ import annotations

import numpy as np
import pytest

from emokit.features.base import (
    BaseTransform,
    FeaturePipeline,
    TransformRegistry,
)
from emokit.features.eeg import BandpowerExtractor, DEExtractor, EEGNormalizer
from emokit.features.peripheral import (
    GSRExtractor,
    HRVExtractor,
    ModalityFusionTransform,
)
from emokit.utils import EmoKitFeatureError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ScaleTransform(BaseTransform):
    """Multiply every element by a fixed scalar (for pipeline testing)."""

    def __init__(self, factor: float = 2.0) -> None:
        self.factor = factor

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> _ScaleTransform:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.factor


class _OffsetTransform(BaseTransform):
    """Add a fixed offset (for pipeline testing)."""

    def __init__(self, offset: float = 1.0) -> None:
        self.offset = offset

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> _OffsetTransform:
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X + self.offset


def _make_sine(freq_hz: float, fs: int, duration: float) -> np.ndarray:
    """Return a pure sine wave of shape ``(T,)``."""
    t = np.arange(int(fs * duration)) / fs
    return np.sin(2.0 * np.pi * freq_hz * t)


# ---------------------------------------------------------------------------
# TransformRegistry
# ---------------------------------------------------------------------------


class TestTransformRegistry:
    def test_register_and_retrieve(self) -> None:
        reg = TransformRegistry()

        @reg.register("dummy")
        class Dummy(BaseTransform):
            def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> Dummy:
                return self

            def transform(self, X: np.ndarray) -> np.ndarray:
                return X

        assert reg["dummy"] is Dummy
        assert "dummy" in reg

    def test_duplicate_raises(self) -> None:
        reg = TransformRegistry()

        @reg.register("dup")
        class A(BaseTransform):
            def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> A:
                return self

            def transform(self, X: np.ndarray) -> np.ndarray:
                return X

        with pytest.raises(EmoKitFeatureError, match="already registered"):

            @reg.register("dup")
            class B(BaseTransform):
                def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> B:
                    return self

                def transform(self, X: np.ndarray) -> np.ndarray:
                    return X

    def test_missing_key_raises(self) -> None:
        reg = TransformRegistry()
        with pytest.raises(EmoKitFeatureError, match="not found"):
            reg["nonexistent"]


# ---------------------------------------------------------------------------
# FeaturePipeline
# ---------------------------------------------------------------------------


class TestFeaturePipeline:
    def test_fit_transform_sequence(self) -> None:
        pipe = FeaturePipeline(
            [
                ("scale", _ScaleTransform(factor=3.0)),
                ("offset", _OffsetTransform(offset=10.0)),
            ]
        )
        X = np.array([1.0, 2.0, 3.0])
        result = pipe.fit_transform(X)
        np.testing.assert_allclose(result, X * 3.0 + 10.0)

    def test_yaml_round_trip(self) -> None:
        pipe = FeaturePipeline(
            [
                ("scale", _ScaleTransform(factor=5.0)),
                ("offset", _OffsetTransform(offset=-1.0)),
            ]
        )
        yaml_str = pipe.to_yaml()
        assert "pipeline" in yaml_str
        assert "scale" in yaml_str

    def test_from_config(self) -> None:
        config = {
            "pipeline": [
                {"name": "de", "transform": "DEExtractor", "params": {"fs": 128}},
                {"name": "norm", "transform": "EEGNormalizer"},
            ]
        }
        pipe = FeaturePipeline.from_config(config)
        assert len(pipe.steps) == 2
        assert pipe.steps[0][0] == "de"
        assert isinstance(pipe.steps[0][1], DEExtractor)
        assert isinstance(pipe.steps[1][1], EEGNormalizer)

    def test_from_config_unknown_transform(self) -> None:
        config = {
            "pipeline": [
                {"name": "bad", "transform": "NoSuchTransform"},
            ]
        }
        with pytest.raises(EmoKitFeatureError, match="Unknown transform"):
            FeaturePipeline.from_config(config)

    def test_duplicate_step_names_raises(self) -> None:
        with pytest.raises(EmoKitFeatureError, match="unique"):
            FeaturePipeline(
                [
                    ("same", _ScaleTransform()),
                    ("same", _OffsetTransform()),
                ]
            )


# ---------------------------------------------------------------------------
# DEExtractor
# ---------------------------------------------------------------------------


class TestDEExtractor:
    def test_shape(self) -> None:
        N, C, T = 4, 3, 256
        X = np.random.randn(N, C, T)
        de = DEExtractor(fs=128).fit_transform(X)
        assert de.shape == (N, C, 5)

    def test_alpha_dominant_for_10hz_sine(self) -> None:
        """A pure 10 Hz sine should have its highest DE in the alpha band."""
        fs = 128
        duration = 2.0
        sine = _make_sine(10.0, fs, duration)
        X = sine[np.newaxis, np.newaxis, :]  # (1, 1, T)

        de = DEExtractor(fs=fs).fit_transform(X)
        alpha_idx = 2  # delta, theta, alpha, beta, gamma
        assert (
            de[0, 0, alpha_idx] == de[0, 0, :].max()
        ), f"Alpha DE should be max; got DE = {de[0, 0, :]}"

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(AssertionError, match="Expected \\(N,C,T\\)"):
            DEExtractor().transform(np.zeros((10, 5)))

    def test_single_sample(self) -> None:
        X = np.random.randn(1, 1, 128)
        de = DEExtractor(fs=128).fit_transform(X)
        assert de.shape == (1, 1, 5)


# ---------------------------------------------------------------------------
# BandpowerExtractor
# ---------------------------------------------------------------------------


class TestBandpowerExtractor:
    def test_shape(self) -> None:
        N, C, T = 8, 4, 512
        X = np.random.randn(N, C, T)
        bp = BandpowerExtractor(fs=256).fit_transform(X)
        assert bp.shape == (N, C, 5)

    def test_nonnegative(self) -> None:
        X = np.random.randn(4, 2, 256)
        bp = BandpowerExtractor(fs=128).fit_transform(X)
        assert np.all(bp >= 0)

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(AssertionError, match="3-D"):
            BandpowerExtractor().transform(np.zeros((10,)))


# ---------------------------------------------------------------------------
# EEGNormalizer
# ---------------------------------------------------------------------------


class TestEEGNormalizer:
    def test_zero_mean_unit_var(self) -> None:
        rng = np.random.default_rng(42)
        X_train = rng.normal(loc=5.0, scale=3.0, size=(100, 4, 32))
        norm = EEGNormalizer()
        X_out = norm.fit_transform(X_train)
        np.testing.assert_allclose(X_out.mean(axis=0), 0.0, atol=1e-6)
        np.testing.assert_allclose(X_out.std(axis=0), 1.0, atol=1e-2)

    def test_transform_without_fit_raises(self) -> None:
        with pytest.raises(EmoKitFeatureError, match="not been fitted"):
            EEGNormalizer().transform(np.zeros((2, 3, 4)))

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(AssertionError, match="3-D"):
            EEGNormalizer().fit(np.zeros((10, 5)))


# ---------------------------------------------------------------------------
# HRVExtractor
# ---------------------------------------------------------------------------


class TestHRVExtractor:
    def test_shape_and_rmssd_positive(self) -> None:
        """Toy ECG-like signal: periodic peaks to simulate QRS."""
        fs = 256
        duration = 10.0
        t = np.arange(int(fs * duration)) / fs
        ecg_like = np.sin(2 * np.pi * 1.2 * t)
        ecg_like += 0.5 * np.sin(2 * np.pi * 12.0 * t)
        X = ecg_like[np.newaxis, np.newaxis, :]

        hrv = HRVExtractor(fs=fs).fit_transform(X)
        assert hrv.shape == (1, 5)
        if not np.isnan(hrv[0, 0]):
            assert hrv[0, 0] > 0, "RMSSD should be positive when valid"

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(AssertionError, match="\\(N, 1, T\\)"):
            HRVExtractor().transform(np.zeros((5, 3, 100)))


# ---------------------------------------------------------------------------
# GSRExtractor
# ---------------------------------------------------------------------------


class TestGSRExtractor:
    def test_shape(self) -> None:
        rng = np.random.default_rng(99)
        X = rng.normal(size=(3, 1, 512)).clip(0)
        gsr = GSRExtractor(fs=128).fit_transform(X)
        assert gsr.shape == (3, 3)

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(AssertionError, match="\\(N, 1, T\\)"):
            GSRExtractor().transform(np.zeros((2, 4, 100)))


# ---------------------------------------------------------------------------
# ModalityFusionTransform
# ---------------------------------------------------------------------------


class TestModalityFusionTransform:
    def test_concatenation(self) -> None:
        eeg = np.ones((10, 4, 5))
        gsr = np.ones((10, 3))
        data = {"eeg": eeg, "gsr": gsr}

        fuse = ModalityFusionTransform()
        out = fuse.fit_transform(data)
        assert out.shape == (10, 4 * 5 + 3)

    def test_explicit_order(self) -> None:
        a = np.zeros((2, 3))
        b = np.ones((2, 4))
        fuse = ModalityFusionTransform(modality_order=["b", "a"])
        out = fuse.fit_transform({"a": a, "b": b})
        assert out.shape == (2, 7)
        np.testing.assert_allclose(out[:, :4], 1.0)
        np.testing.assert_allclose(out[:, 4:], 0.0)

    def test_not_dict_raises(self) -> None:
        with pytest.raises(AssertionError, match="dict"):
            ModalityFusionTransform().fit(np.zeros((3, 5)))

    def test_transform_without_fit_raises(self) -> None:
        with pytest.raises(EmoKitFeatureError, match="not been fitted"):
            ModalityFusionTransform().transform({"a": np.zeros((2, 3))})

    def test_missing_modality_raises(self) -> None:
        fuse = ModalityFusionTransform()
        fuse.fit({"a": np.zeros((2, 3))})
        with pytest.raises(EmoKitFeatureError, match="Missing modality"):
            fuse.transform({"b": np.zeros((2, 3))})


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_array_de(self) -> None:
        X = np.empty((0, 2, 128))
        de = DEExtractor(fs=128).fit_transform(X)
        assert de.shape == (0, 2, 5)

    def test_empty_array_bandpower(self) -> None:
        X = np.empty((0, 2, 128))
        bp = BandpowerExtractor(fs=128).fit_transform(X)
        assert bp.shape == (0, 2, 5)

    def test_single_sample_bandpower(self) -> None:
        X = np.random.randn(1, 1, 128)
        bp = BandpowerExtractor(fs=128).fit_transform(X)
        assert bp.shape == (1, 1, 5)

    def test_wrong_shape_de_2d(self) -> None:
        with pytest.raises(AssertionError):
            DEExtractor().transform(np.zeros((10, 5)))

    def test_wrong_shape_normalizer_2d(self) -> None:
        with pytest.raises(AssertionError):
            EEGNormalizer().fit(np.zeros((10, 5)))


# ---------------------------------------------------------------------------
# Paper-aligned tests (P0-1)
# ---------------------------------------------------------------------------


def test_de_alpha_dominates_for_10hz_sine():
    """Pure 10Hz sine -> alpha band DE must be strictly highest."""
    fs, T = 128, 512
    t = np.linspace(0, T / fs, T, endpoint=False)
    sig = np.sin(2 * np.pi * 10 * t).astype(np.float32)
    X = sig[np.newaxis, np.newaxis, :]  # (1, 1, 512)
    de = DEExtractor(fs=fs).transform(X)  # (1, 1, 5)
    assert de.shape == (1, 1, 5)
    alpha_idx = 2
    assert de[0, 0, alpha_idx] == de[0, 0].max(), f"Alpha not dominant: {de[0, 0]}"


def test_de_output_dtype_and_shape():
    X = np.random.randn(16, 32, 512).astype(np.float32)
    de = DEExtractor(fs=128).transform(X)
    assert de.shape == (16, 32, 5)
    assert de.dtype == np.float32


def test_de_stateless_no_fit_needed():
    X = np.random.randn(4, 32, 512).astype(np.float32)
    ext = DEExtractor(fs=128)
    de1 = ext.transform(X)
    de2 = ext.fit_transform(X)
    np.testing.assert_array_equal(de1, de2)
