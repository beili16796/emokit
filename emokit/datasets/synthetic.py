# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Synthetic dataset for testing, quick demos, and CI pipelines."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from emokit.datasets.base import BaseDataset, _REGISTRY

logger = logging.getLogger(__name__)


@_REGISTRY.register("SYNTHETIC")
class SyntheticDataset(BaseDataset):
    """Generates reproducible synthetic EEG-like data in memory.

    Useful for unit tests, quick demos, and CI pipelines where real data
    is unavailable.

    Args:
        n_subjects: Number of virtual subjects.
        n_trials: Trials per subject.
        n_channels: Number of EEG channels.
        fs: Sampling frequency in Hz.
        window_sec: Window length in seconds (forwarded to base class).
        n_classes: Number of label classes.
        seed: Base random seed (combined with subject_id for per-subject RNG).
        root: Ignored — kept for interface compatibility.
        subjects: Optional subset of subject IDs to load.
        overlap: Sliding-window overlap fraction.
        modalities: Signal modalities to include.
    """

    def __init__(
        self,
        n_subjects: int = 5,
        n_trials: int = 20,
        n_channels: int = 32,
        fs: float = 128.0,
        window_sec: float = 1.0,
        n_classes: int = 2,
        seed: int = 42,
        root: str | None = None,
        subjects: list[int] | None = None,
        overlap: float = 0.0,
        modalities: list[str] | None = None,
    ) -> None:
        super().__init__(
            root=root or "/tmp/emokit_synthetic",
            subjects=subjects,
            window_sec=window_sec,
            overlap=overlap,
            modalities=modalities,
        )
        self.n_subjects = n_subjects
        self.n_trials = n_trials
        self.n_channels = n_channels
        self.fs = fs
        self.n_classes = n_classes
        self.seed = seed

    def _get_fs(self) -> float:
        return self.fs

    def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
        """Generate synthetic data for one subject.

        Args:
            subject_id: 1-based subject identifier.

        Returns:
            Dict with ``'eeg'`` array of shape ``(n_trials, n_channels,
            n_samples)`` and ``'labels'`` of shape ``(n_trials,)``.
        """
        rng = np.random.default_rng(self.seed + subject_id)
        n_samples = int(round(self.window_sec * self.fs))
        eeg = rng.standard_normal((self.n_trials, self.n_channels, n_samples))
        labels = rng.integers(0, self.n_classes, size=self.n_trials)
        return {"eeg": eeg, "labels": labels}

    def get_subject_ids(self) -> list[int]:
        """Return list of subject IDs from 1 to *n_subjects*."""
        return list(range(1, self.n_subjects + 1))

    def get_channel_names(self, modality: str) -> list[str]:
        """Return generic channel names."""
        return [f"SYN{i}" for i in range(self.n_channels)]

    def get_label_names(self) -> list[str]:
        """Return generic label names."""
        return [f"class_{i}" for i in range(self.n_classes)]
