# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Peripheral physiological signal feature extraction (ECG / GSR)."""

from __future__ import annotations

import logging

import numpy as np

from emokit.features.base import GLOBAL_REGISTRY, BaseTransform
from emokit.utils import EmoKitFeatureError

logger = logging.getLogger(__name__)


@GLOBAL_REGISTRY.register("HRVExtractor")
class HRVExtractor(BaseTransform):
    """Heart-rate variability feature extractor from single-channel ECG.

    Extracts five HRV metrics per window: RMSSD, SDNN, LF power, HF power,
    and LF/HF ratio.  Uses `neurokit2` for R-peak detection and HRV analysis.

    Args:
        fs: ECG sampling rate in Hz.
    """

    def __init__(self, fs: int = 256) -> None:
        self.fs = fs

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> HRVExtractor:
        """No-op; HRV extraction is stateless.

        Args:
            X: ECG array of shape ``(N, 1, T)``.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract HRV features from ECG windows.

        Args:
            X: ECG array of shape ``(N, 1, T)``.

        Returns:
            HRV feature matrix of shape ``(N, 5)`` with columns
            ``[RMSSD, SDNN, LF, HF, LF/HF]``.  Failed windows are
            filled with ``NaN``.

        Raises:
            ValueError: If *X* does not have shape ``(N, 1, T)``.
        """
        import neurokit2 as nk

        assert X.ndim == 3 and X.shape[1] == 1, (
            f"HRVExtractor expects shape (N, 1, T), got {X.shape}"
        )

        n = X.shape[0]
        out = np.full((n, 5), np.nan, dtype=np.float64)

        for i in range(n):
            try:
                signal = X[i, 0, :]
                processed, info = nk.ecg_process(signal, sampling_rate=self.fs)
                rpeaks = info["ECG_R_Peaks"]

                if len(rpeaks) < 3:
                    logger.debug("Window %d: too few R-peaks (%d)", i, len(rpeaks))
                    continue

                rri = np.diff(rpeaks) / self.fs * 1000.0  # ms
                rmssd = np.sqrt(np.mean(np.diff(rri) ** 2))
                sdnn = np.std(rri, ddof=1)

                try:
                    hrv_freq = nk.hrv_frequency(
                        rpeaks, sampling_rate=self.fs, show=False
                    )
                    lf = float(hrv_freq["HRV_LF"].iloc[0])
                    hf = float(hrv_freq["HRV_HF"].iloc[0])
                    lf_hf = lf / hf if hf > 0 else np.nan
                except Exception:
                    lf, hf, lf_hf = np.nan, np.nan, np.nan

                out[i] = [rmssd, sdnn, lf, hf, lf_hf]
            except Exception:
                logger.debug("Window %d: ECG processing failed", i, exc_info=True)

        return out


@GLOBAL_REGISTRY.register("GSRExtractor")
class GSRExtractor(BaseTransform):
    """Galvanic skin response (EDA) feature extractor.

    Decomposes the EDA signal into tonic (SCL) and phasic (SCR) components
    via `neurokit2` and extracts three features per window: tonic mean,
    phasic peak count, and mean phasic peak amplitude.

    Args:
        fs: EDA sampling rate in Hz.
    """

    def __init__(self, fs: int = 128) -> None:
        self.fs = fs

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> GSRExtractor:
        """No-op; GSR extraction is stateless.

        Args:
            X: EDA array of shape ``(N, 1, T)``.
            y: Ignored.

        Returns:
            self
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract GSR features from EDA windows.

        Args:
            X: EDA array of shape ``(N, 1, T)``.

        Returns:
            GSR feature matrix of shape ``(N, 3)`` with columns
            ``[tonic_mean, phasic_peak_count, phasic_peak_mean_amp]``.
            Failed windows are filled with ``NaN``.

        Raises:
            ValueError: If *X* does not have shape ``(N, 1, T)``.
        """
        import neurokit2 as nk

        assert X.ndim == 3 and X.shape[1] == 1, (
            f"GSRExtractor expects shape (N, 1, T), got {X.shape}"
        )

        n = X.shape[0]
        out = np.full((n, 3), np.nan, dtype=np.float64)

        for i in range(n):
            try:
                signal = X[i, 0, :]
                processed, info = nk.eda_process(signal, sampling_rate=self.fs)

                tonic = processed["EDA_Tonic"].values
                tonic_mean = float(np.mean(tonic))

                peaks_mask = processed["SCR_Peaks"].values.astype(bool)
                peak_count = float(np.sum(peaks_mask))

                if peak_count > 0:
                    amplitudes = processed["SCR_Amplitude"].values[peaks_mask]
                    peak_amp = float(np.nanmean(amplitudes))
                else:
                    peak_amp = 0.0

                out[i] = [tonic_mean, peak_count, peak_amp]
            except Exception:
                logger.debug("Window %d: EDA processing failed", i, exc_info=True)

        return out


@GLOBAL_REGISTRY.register("ModalityFusionTransform")
class ModalityFusionTransform(BaseTransform):
    """Late-fusion transform that concatenates feature arrays from multiple modalities.

    Unlike other transforms, :meth:`fit` and :meth:`transform` accept a
    *dict* mapping modality names to their respective feature arrays rather
    than a single ``np.ndarray``.

    Args:
        modality_order: Optional explicit ordering of modality keys.  When
            *None*, keys are sorted alphabetically for reproducibility.
    """

    def __init__(self, modality_order: list[str] | None = None) -> None:
        self.modality_order = modality_order
        self._fitted_order: list[str] | None = None

    def fit(  # type: ignore[override]
        self,
        X: dict[str, np.ndarray],
        y: np.ndarray | None = None,
    ) -> ModalityFusionTransform:
        """Record modality ordering from training data.

        Args:
            X: Dict mapping modality name to feature array.
            y: Ignored.

        Returns:
            self

        Raises:
            ValueError: If *X* is not a dict.
        """
        assert isinstance(X, dict), (
            f"ModalityFusionTransform expects dict input, got {type(X).__name__}"
        )
        if self.modality_order is not None:
            self._fitted_order = list(self.modality_order)
        else:
            self._fitted_order = sorted(X.keys())
        return self

    def transform(  # type: ignore[override]
        self, X: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Flatten and concatenate features across modalities.

        Each per-modality array is reshaped to ``(N, -1)`` before
        concatenation along axis 1.

        Args:
            X: Dict mapping modality name to feature array.

        Returns:
            Concatenated feature matrix of shape ``(N, total_features)``.

        Raises:
            EmoKitFeatureError: If :meth:`fit` has not been called or a
                required modality key is missing.
            ValueError: If *X* is not a dict.
        """
        assert isinstance(X, dict), (
            f"ModalityFusionTransform expects dict input, got {type(X).__name__}"
        )
        if self._fitted_order is None:
            raise EmoKitFeatureError("ModalityFusionTransform has not been fitted.")

        arrays: list[np.ndarray] = []
        for key in self._fitted_order:
            if key not in X:
                raise EmoKitFeatureError(f"Missing modality '{key}' in input dict.")
            arr = X[key]
            arrays.append(arr.reshape(arr.shape[0], -1))

        return np.concatenate(arrays, axis=1)

    def fit_transform(  # type: ignore[override]
        self,
        X: dict[str, np.ndarray],
        y: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fit and transform in one step.

        Args:
            X: Dict mapping modality name to feature array.
            y: Ignored.

        Returns:
            Concatenated feature matrix.
        """
        return self.fit(X, y).transform(X)
