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

    With labels (the normal training-pipeline path), mixup perturbs selected
    samples in place so the feature and label arrays remain aligned.  Without
    labels, it appends synthetic samples for standalone augmentation probes.

    Args:
        alpha: Beta distribution parameter controlling interpolation strength.
        ratio: Fraction of synthetic samples to append when ``y`` is omitted.
        p: Probability of in-place mixup per sample when ``y`` is provided.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        ratio: float | None = None,
        p: float | None = None,
        seed: int = 42,
    ) -> None:
        self.alpha = alpha
        self.ratio = 0.5 if ratio is None else ratio
        self.p = self.ratio if p is None else p
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
        if not self._is_training or X.shape[0] == 0 or self.alpha <= 0:
            return X

        rng = np.random.default_rng(self.seed)
        n = X.shape[0]

        if y is not None:
            y_arr = np.asarray(y)
            X_out = X.copy()
            for i in range(n):
                if rng.random() > self.p:
                    continue
                candidates = np.where(y_arr == y_arr[i])[0]
                if len(candidates) < 2:
                    candidates = np.arange(n)
                j = int(rng.choice(candidates))
                lam = float(rng.beta(self.alpha, self.alpha))
                X_out[i] = lam * X[i] + (1.0 - lam) * X[j]
            return X_out.astype(X.dtype)

        if self.ratio <= 0:
            return X
        n_aug = max(1, int(n * self.ratio))
        lam = rng.beta(self.alpha, self.alpha, size=n_aug)
        idx_a = rng.integers(0, n, size=n_aug)
        idx_b = rng.integers(0, n, size=n_aug)
        shape = (n_aug,) + (1,) * (X.ndim - 1)
        X_aug = lam.reshape(shape) * X[idx_a] + (1 - lam).reshape(shape) * X[idx_b]
        return np.concatenate([X, X_aug], axis=0).astype(X.dtype)


@GLOBAL_REGISTRY.register("TemporalSegmentPermutation")
class TemporalSegmentPermutation(BaseTransform):
    """Temporal segment permutation augmentation for EEG windows.

    Args:
        n_segments: Number of temporal segments to split into.
        ratio: Fraction of samples to append when ``y`` is omitted.
        p: Probability of in-place permutation per sample when ``y`` is provided.
        seed: Random seed.
    """

    def __init__(
        self,
        n_segments: int = 4,
        ratio: float | None = None,
        p: float | None = None,
        seed: int = 42,
    ) -> None:
        self.n_segments = n_segments
        self.ratio = 0.5 if ratio is None else ratio
        self.p = self.ratio if p is None else p
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

    def _permute_one(self, sample: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        T = sample.shape[-1]
        seg_len = T // self.n_segments
        if seg_len < 1:
            return sample
        segments = [
            sample[..., i * seg_len : (i + 1) * seg_len] for i in range(self.n_segments)
        ]
        remainder = sample[..., self.n_segments * seg_len :]
        order = rng.permutation(self.n_segments)
        shuffled = [segments[i] for i in order]
        if remainder.shape[-1] > 0:
            shuffled.append(remainder)
        return np.concatenate(shuffled, axis=-1)

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        self.fit(X, y)
        if not self._is_training or X.ndim != 3:
            return X

        rng = np.random.default_rng(self.seed)
        n = X.shape[0]

        if y is not None:
            X_out = X.copy()
            for i in range(n):
                if rng.random() <= self.p:
                    X_out[i] = self._permute_one(X_out[i], rng)
            return X_out.astype(X.dtype)

        if self.ratio <= 0:
            return X
        n_aug = max(1, int(n * self.ratio))
        indices = rng.choice(n, size=n_aug, replace=True)
        X_aug = np.stack([self._permute_one(X[idx].copy(), rng) for idx in indices])
        return np.concatenate([X, X_aug], axis=0).astype(X.dtype)
