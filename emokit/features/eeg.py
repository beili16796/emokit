# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""EEG-specific feature extraction transforms."""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from emokit.features.base import GLOBAL_REGISTRY, BaseTransform
from emokit.utils import EmoKitFeatureError

logger = logging.getLogger(__name__)

DEFAULT_BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


@GLOBAL_REGISTRY.register("DEExtractor")
class DEExtractor(BaseTransform):
    """Differential Entropy extractor for EEG signals.

    Strictly follows Zheng et al. (2015): for a Gaussian-distributed signal,
    DE = 0.5 * log(2*pi*e * sigma^2). Since 2*pi*e is a constant offset
    irrelevant for cross-band comparison, we compute
    ``DE_band = log(var(bandpass_filtered_signal))``.

    The signal is bandpass-filtered per band using a zero-phase Butterworth
    filter (``scipy.signal.sosfiltfilt``), then log-variance is taken over the
    time axis.

    Args:
        fs: Sampling frequency in Hz.
        bands: Mapping of band name to ``(low_hz, high_hz)``.
        filter_order: Butterworth filter order.
    """

    def __init__(
        self,
        fs: int = 128,
        bands: dict[str, tuple[float, float]] | None = None,
        filter_order: int = 5,
    ) -> None:
        self.fs = fs
        self.bands = bands if bands is not None else dict(DEFAULT_BANDS)
        self.filter_order = filter_order
        self._sos: dict[str, np.ndarray] = {}
        for name, (lo, hi) in self.bands.items():
            self._sos[name] = butter(
                self.filter_order, [lo, hi], btype="bandpass", fs=self.fs, output="sos"
            )

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> DEExtractor:
        """No-op; DE extraction is stateless.

        Args:
            X: EEG array of shape ``(N, C, T)``.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute DE features as log-variance of bandpass-filtered signals.

        Args:
            X: EEG array of shape ``(N, C, T)``.

        Returns:
            DE features of shape ``(N, C, num_bands)``.

        Raises:
            ValueError: If *X* is not 3-D.
        """
        assert X.ndim == 3, f"Expected (N,C,T), got {X.shape}"
        n, c, _t = X.shape
        n_bands = len(self.bands)
        de = np.empty((n, c, n_bands), dtype=np.float32)

        for b_idx, (band_name, sos) in enumerate(
            zip(self.bands, self._sos.values())
        ):
            filtered = sosfiltfilt(sos, X, axis=-1)
            de[:, :, b_idx] = np.log(np.var(filtered, axis=-1) + 1e-8)

        return de


@GLOBAL_REGISTRY.register("BandpowerExtractor")
class BandpowerExtractor(BaseTransform):
    """Absolute band-power extractor via Welch's method.

    Args:
        fs: Sampling frequency in Hz.
        bands: Mapping of band name to ``(low_hz, high_hz)``.
    """

    def __init__(
        self,
        fs: int = 128,
        bands: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.fs = fs
        self.bands = bands if bands is not None else dict(DEFAULT_BANDS)

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> BandpowerExtractor:
        """No-op; bandpower does not require fitting.

        Args:
            X: EEG array of shape ``(N, C, T)``.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Compute absolute band power.

        Args:
            X: EEG array of shape ``(N, C, T)``.

        Returns:
            Band-power features of shape ``(N, C, num_bands)``.

        Raises:
            ValueError: If *X* is not 3-D.
        """
        assert X.ndim == 3, (
            f"BandpowerExtractor expects 3-D input (N,C,T), got {X.ndim}-D"
        )

        n, c, t = X.shape
        num_bands = len(self.bands)
        freqs, psd = welch(X, fs=self.fs, nperseg=min(t, 256), axis=-1)
        freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

        bp = np.empty((n, c, num_bands), dtype=np.float64)
        for i, (_, (lo, hi)) in enumerate(self.bands.items()):
            mask = (freqs >= lo) & (freqs < hi)
            if not np.any(mask):
                bp[:, :, i] = 0.0
                continue
            bp[:, :, i] = np.sum(psd[..., mask], axis=-1) * freq_res

        return bp


@GLOBAL_REGISTRY.register("EEGNormalizer")
class EEGNormalizer(BaseTransform):
    """Per-channel z-score normalizer for EEG features.

    Handles both raw EEG ``(N, C, T)`` and extracted features ``(N, C, F)``.
    Statistics are computed along the first axis (samples).

    Args:
        eps: Small constant to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> EEGNormalizer:
        """Compute per-channel mean and std from training data.

        Args:
            X: Array of shape ``(N, C, T)`` or ``(N, C, F)``.
            y: Ignored.

        Returns:
            self

        Raises:
            ValueError: If *X* is not 3-D.
        """
        assert X.ndim == 3, (
            f"EEGNormalizer expects 3-D input, got {X.ndim}-D"
        )
        self._mean = np.mean(X, axis=0, keepdims=True)
        self._std = np.std(X, axis=0, keepdims=True)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalization using stored statistics.

        Args:
            X: Array of shape ``(N, C, T)`` or ``(N, C, F)``.

        Returns:
            Normalized array with same shape.

        Raises:
            EmoKitFeatureError: If :meth:`fit` has not been called.
            ValueError: If *X* is not 3-D.
        """
        if self._mean is None or self._std is None:
            raise EmoKitFeatureError(
                "EEGNormalizer has not been fitted. Call fit() first."
            )
        assert X.ndim == 3, (
            f"EEGNormalizer expects 3-D input, got {X.ndim}-D"
        )
        return (X - self._mean) / (self._std + self.eps)
