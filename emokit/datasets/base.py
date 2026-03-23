# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Base dataset abstractions, registry, and windowing utilities."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from emokit.utils import EmoKitDataError, get_data_root

logger = logging.getLogger(__name__)


def segment_trials(
    data: np.ndarray,
    fs: float,
    window_sec: float = 4.0,
    overlap: float = 0.5,
) -> np.ndarray:
    """Slide a fixed-length window over each trial and stack the result.

    Args:
        data: Array of shape ``(n_trials, n_channels, n_samples)``.
        fs: Sampling frequency in Hz.
        window_sec: Window length in seconds.
        overlap: Fractional overlap in ``[0, 1)``.

    Returns:
        Array of shape ``(n_windows, n_channels, n_samples_per_window)``.

    Raises:
        ValueError: If *data* is not 3-D or *overlap* is out of range.
    """
    assert data.ndim == 3, f"Expected 3D array (N,C,T), got {data.shape}"
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")

    n_trials, n_channels, n_total = data.shape
    win_samples = int(round(window_sec * fs))
    step = max(1, int(round(win_samples * (1.0 - overlap))))

    if win_samples > n_total:
        raise ValueError(
            f"Window ({win_samples} samples) exceeds trial length ({n_total} samples)"
        )

    windows: list[np.ndarray] = []
    for trial_idx in range(n_trials):
        start = 0
        while start + win_samples <= n_total:
            windows.append(data[trial_idx, :, start : start + win_samples])
            start += step

    if len(windows) == 0:
        return np.empty((0, n_channels, win_samples), dtype=data.dtype)

    return np.stack(windows, axis=0)


class DatasetRegistry(dict):
    """Name → class mapping with a ``register`` decorator.

    Example::

        registry = DatasetRegistry()

        @registry.register("MY_DATASET")
        class MyDataset(BaseDataset):
            ...

        ds = registry["MY_DATASET"](root="/data")
    """

    def register(self, name: str):
        """Return a decorator that adds *cls* under *name*.

        Args:
            name: Canonical dataset identifier (e.g. ``'DEAP'``).
        """
        def decorator(cls: type) -> type:
            if name in self:
                logger.warning("Overwriting registry entry '%s'", name)
            self[name] = cls
            logger.debug("Registered dataset '%s' -> %s", name, cls.__name__)
            return cls
        return decorator

    def available(self) -> list[str]:
        """Return sorted list of registered dataset names."""
        return sorted(self.keys())


_REGISTRY = DatasetRegistry()


def load_dataset(name: str, **kwargs: Any) -> BaseDataset:
    """Instantiate a registered dataset by name.

    Args:
        name: Registry key (e.g. ``'DEAP'``, ``'SEED'``).
        **kwargs: Forwarded to the dataset constructor.

    Returns:
        An instance of the requested :class:`BaseDataset` subclass.

    Raises:
        EmoKitDataError: If *name* is not in the registry.
    """
    if name not in _REGISTRY:
        raise EmoKitDataError(
            f"Unknown dataset '{name}'. Available: {_REGISTRY.available()}"
        )
    return _REGISTRY[name](**kwargs)


class BaseDataset(ABC):
    """Abstract base class for all EmoKit datasets.

    Subclasses must implement :meth:`read_raw`, :meth:`get_subject_ids`,
    :meth:`get_channel_names`, and :meth:`get_label_names`.

    Args:
        root: Path to the dataset's root directory.
        subjects: Subset of subject IDs to load; ``None`` loads all.
        window_sec: Sliding-window length in seconds.
        overlap: Fractional overlap for the sliding window.
        modalities: Which signal modalities to include (e.g. ``['eeg']``).
    """

    def __init__(
        self,
        root: str | None = None,
        subjects: list[int] | None = None,
        window_sec: float = 4.0,
        overlap: float = 0.5,
        modalities: list[str] | None = None,
    ) -> None:
        self.root = Path(root) if root is not None else get_data_root()
        self.subjects = subjects
        self.window_sec = window_sec
        self.overlap = overlap
        self.modalities = modalities
        logger.info(
            "Initialised %s  root=%s  subjects=%s",
            self.__class__.__name__,
            self.root,
            self.subjects,
        )

    @abstractmethod
    def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
        """Load raw data for a single subject.

        Args:
            subject_id: 1-based subject identifier.

        Returns:
            Mapping from modality name to array of shape
            ``(n_trials, n_channels, n_samples)``.
        """

    @abstractmethod
    def get_subject_ids(self) -> list[int]:
        """Return the full list of available subject IDs."""

    @abstractmethod
    def get_channel_names(self, modality: str) -> list[str]:
        """Return ordered channel names for *modality*."""

    @abstractmethod
    def get_label_names(self) -> list[str]:
        """Return human-readable label class names."""

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """Load, preprocess, segment, and concatenate data across subjects.

        Returns:
            ``(X, y)`` where *X* has shape ``(n_windows, n_channels, n_samples)``
            and *y* has shape ``(n_windows,)``.
        """
        subject_ids = self.subjects if self.subjects else self.get_subject_ids()
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []

        for sid in subject_ids:
            logger.info("Loading subject %d …", sid)
            raw = self.read_raw(sid)

            modalities = self.modalities or [
                k for k in raw if k != "labels"
            ]
            arrays = [raw[m] for m in modalities if m in raw and m != "labels"]
            if not arrays:
                logger.warning("No matching modalities for subject %d, skipping", sid)
                continue

            data = np.concatenate(arrays, axis=1)
            assert data.ndim == 3, f"Expected 3D array (N,C,T), got {data.shape}"

            labels = raw.get("labels")
            if labels is None:
                raise EmoKitDataError(
                    f"read_raw for subject {sid} must include 'labels' key"
                )

            windows = segment_trials(
                data, self._get_fs(), self.window_sec, self.overlap,
            )

            n_trials = data.shape[0]
            n_total = data.shape[2]
            win_samples = int(round(self.window_sec * self._get_fs()))
            step = max(1, int(round(win_samples * (1.0 - self.overlap))))
            wins_per_trial = max(
                0,
                (n_total - win_samples) // step + 1,
            )
            trial_labels = np.repeat(labels, wins_per_trial)

            all_x.append(windows)
            all_y.append(trial_labels)

        if not all_x:
            raise EmoKitDataError("No data loaded for any subject")

        X = np.concatenate(all_x, axis=0)
        y = np.concatenate(all_y, axis=0)
        logger.info("Loaded X=%s  y=%s", X.shape, y.shape)
        return X, y

    def _get_fs(self) -> float:
        """Return the (post-preprocessing) sampling rate in Hz.

        Subclasses should override if the effective rate differs from the
        raw rate.
        """
        raise NotImplementedError("Subclass must implement _get_fs()")
