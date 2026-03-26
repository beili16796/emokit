# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Physiological-signal data augmentation for DE feature arrays.

Implements augmentation strategies suitable for EEG differential entropy
features, following recent wearable affective computing literature.

Usage in a YAML config::

    feature_pipeline:
      steps:
        - name: DEExtractor
          params: {fs: 128}
        - name: EEGNormalizer
          params: {}
        - name: FeatureMixup
          params: {alpha: 0.2}
"""

from __future__ import annotations

import logging

import numpy as np

from emokit.features.base import GLOBAL_REGISTRY, BaseTransform

logger = logging.getLogger(__name__)


@GLOBAL_REGISTRY.register("FeatureMixup")
class FeatureMixup(BaseTransform):
    """Feature-level Mixup augmentation for DE features.

    During ``fit_transform`` (training), generates virtual samples by
    convex interpolation of random pairs.  ``transform`` (inference) is
    a no-op.

    Args:
        alpha: Beta distribution parameter controlling interpolation
            strength.  Larger values produce more aggressive mixing.
        ratio: Fraction of additional synthetic samples relative to
            the original batch size.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.alpha = alpha
        self.ratio = ratio
        self.seed = seed
        self._is_training = True

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> FeatureMixup:
        self._is_training = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """No-op at inference time."""
        self._is_training = False
        return X

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        """Apply mixup augmentation during training."""
        self.fit(X, y)
        if not self._is_training or self.ratio <= 0:
            return X

        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        n_aug = max(1, int(n * self.ratio))

        lam = rng.beta(self.alpha, self.alpha, size=n_aug)
        idx_a = rng.integers(0, n, size=n_aug)
        idx_b = rng.integers(0, n, size=n_aug)

        shape = (n_aug,) + (1,) * (X.ndim - 1)
        lam_broad = lam.reshape(shape)
        X_aug = lam_broad * X[idx_a] + (1 - lam_broad) * X[idx_b]

        return np.concatenate([X, X_aug], axis=0).astype(X.dtype)


@GLOBAL_REGISTRY.register("TemporalSegmentPermutation")
class TemporalSegmentPermutation(BaseTransform):
    """Temporal segment permutation augmentation for EEG windows.

    Splits the time axis into ``n_segments`` equal parts and randomly
    permutes them.  This preserves local temporal structure within each
    segment while destroying long-range ordering — a regularisation
    technique for sequence models.

    Expects input shape ``(N, C, T)`` where T is the time dimension.
    At inference (``transform``), acts as a no-op.

    Args:
        n_segments: Number of temporal segments to split into.
        ratio: Fraction of samples to augment (the rest are kept as-is).
        seed: Random seed.
    """

    def __init__(
        self,
        n_segments: int = 4,
        ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        self.n_segments = n_segments
        self.ratio = ratio
        self.seed = seed
        self._is_training = True

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> TemporalSegmentPermutation:
        self._is_training = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._is_training = False
        return X

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(X, y)
        if not self._is_training or X.ndim != 3:
            return X

        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        n_aug = max(1, int(n * self.ratio))
        indices = rng.choice(n, size=n_aug, replace=True)

        T = X.shape[2]
        seg_len = T // self.n_segments
        if seg_len < 1:
            return X

        augmented = []
        for idx in indices:
            sample = X[idx].copy()
            segments = [
                sample[:, i * seg_len : (i + 1) * seg_len]
                for i in range(self.n_segments)
            ]
            remainder = sample[:, self.n_segments * seg_len :]
            rng.shuffle(segments)
            if remainder.shape[1] > 0:
                segments.append(remainder)
            augmented.append(np.concatenate(segments, axis=1))

        X_aug = np.stack(augmented, axis=0)
        return np.concatenate([X, X_aug], axis=0).astype(X.dtype)
