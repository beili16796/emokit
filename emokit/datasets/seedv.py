# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""SEED-V dataset loader (62-channel EEG + 3-channel EOG, 5 emotions)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt

from emokit.datasets.base import _REGISTRY, BaseDataset
from emokit.utils import EmoKitDataError

logger = logging.getLogger(__name__)

_EEG_CHANNELS: list[str] = [
    "Fp1",
    "Fpz",
    "Fp2",
    "AF3",
    "AF4",
    "F7",
    "F5",
    "F3",
    "F1",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "FCz",
    "FC2",
    "FC4",
    "FC6",
    "FT8",
    "T7",
    "C5",
    "C3",
    "C1",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "CPz",
    "CP2",
    "CP4",
    "CP6",
    "TP8",
    "P7",
    "P5",
    "P3",
    "P1",
    "Pz",
    "P2",
    "P4",
    "P6",
    "P8",
    "PO7",
    "PO5",
    "PO3",
    "POz",
    "PO4",
    "PO6",
    "PO8",
    "CB1",
    "O1",
    "Oz",
    "O2",
    "CB2",
]

_EOG_CHANNELS: list[str] = ["hEOG", "vEOG", "hEOG2"]

_EMOTION_LABELS: list[str] = ["happy", "sad", "neutral", "fear", "disgust"]

_N_SUBJECTS: int = 16
_N_SESSIONS: int = 3
_FS: float = 200.0


SEED_62_CHANNELS: list[str] = list(_EEG_CHANNELS)


@_REGISTRY.register("SEED-V")
class SEEDVDataset(BaseDataset):
    """SEED-V: SJTU Emotion EEG Dataset (five emotions).

    Reference:
        Liu et al., *IEEE Trans. Affective Computing*, 2021.

    The dataset is distributed as ``.mat`` files containing either raw EEG
    or pre-extracted differential entropy (DE) features.

    Args:
        root: Path to the SEED-V data directory.
        subjects: Subject IDs to load (1-based, 1–16).
        window_sec: Sliding-window length in seconds.
        overlap: Fractional overlap for the sliding window.
        modalities: Subset of ``{'eeg', 'eog'}`` to include.
        sessions: Which sessions to load (1-based); ``None`` loads all.
        use_de_features: If ``True``, load DE features directly from mat
            instead of raw signals.
    """

    def __init__(
        self,
        root: str | None = None,
        subjects: list[int] | None = None,
        window_sec: float = 4.0,
        overlap: float = 0.5,
        modalities: list[str] | None = None,
        sessions: list[int] | None = None,
        use_de_features: bool = False,
    ) -> None:
        super().__init__(
            root=root,
            subjects=subjects,
            window_sec=window_sec,
            overlap=overlap,
            modalities=modalities,
        )
        self.sessions = sessions or list(range(1, _N_SESSIONS + 1))
        self.use_de_features = use_de_features
        self._pre_extracted: bool = use_de_features

    @property
    def is_pre_extracted(self) -> bool:
        """Whether this dataset contains pre-extracted DE features."""
        return self._pre_extracted

    def _get_fs(self) -> float:
        return _FS

    def get_subject_ids(self) -> list[int]:
        return list(range(1, _N_SUBJECTS + 1))

    def get_channel_names(self, modality: str) -> list[str]:
        if modality == "eeg":
            return list(_EEG_CHANNELS)
        if modality == "eog":
            return list(_EOG_CHANNELS)
        raise EmoKitDataError(f"Unknown SEED-V modality '{modality}'")

    def get_label_names(self) -> list[str]:
        return list(_EMOTION_LABELS)

    def _find_mat(self, subject_id: int, session: int) -> Path:
        """Resolve the ``.mat`` path for a given subject/session pair."""
        patterns = [
            self.root / f"{subject_id}" / f"{session}.mat",
            self.root / f"sub{subject_id}" / f"session{session}.mat",
            self.root / f"s{subject_id:02d}" / f"sess{session:02d}.mat",
            self.root / f"{subject_id}_{session}.mat",
        ]
        for p in patterns:
            if p.exists():
                return p
        raise EmoKitDataError(
            f"No .mat file for subject {subject_id}, session {session} "
            f"under {self.root}. Tried: {[str(p) for p in patterns]}"
        )

    def _load_de_features(self, mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load pre-computed DE features from a ``.mat`` file.

        Returns:
            data: ``(n_trials, n_channels, n_bands)``
            labels: ``(n_trials,)``
        """
        mat = loadmat(str(mat_path), squeeze_me=True)

        de_key = None
        for key in ("de_LDS", "de_lds", "DE_LDS"):
            if key in mat:
                de_key = key
                break
        if de_key is None:
            raise EmoKitDataError(
                f"No DE feature variable found in {mat_path}. "
                f"Available keys: {[k for k in mat if not k.startswith('__')]}"
            )

        de = np.asarray(mat[de_key], dtype=np.float64)
        if de.ndim == 2:
            de = de[np.newaxis, ...]

        labels_raw = mat.get("labels", mat.get("label", None))
        if labels_raw is None:
            raise EmoKitDataError(f"No label variable found in {mat_path}")
        labels = np.asarray(labels_raw, dtype=np.int64).ravel()

        return de, labels

    def _load_raw_eeg(self, mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load raw EEG signals from a ``.mat`` file and band-pass filter.

        Returns:
            data: ``(n_trials, n_channels, n_samples)``
            labels: ``(n_trials,)``
        """
        mat = loadmat(str(mat_path), squeeze_me=True)

        data_keys = [
            k for k in mat if not k.startswith("__") and k != "labels" and k != "label"
        ]
        trials: list[np.ndarray] = []
        for key in sorted(data_keys):
            arr = np.asarray(mat[key], dtype=np.float64)
            if arr.ndim == 2:
                trials.append(arr)

        if not trials:
            raise EmoKitDataError(f"No trial data found in {mat_path}")

        n_eeg = 62
        n_eog = 3
        data_list: list[np.ndarray] = []
        for t in trials:
            if t.shape[0] >= n_eeg + n_eog:
                data_list.append(t[: n_eeg + n_eog, :])
            elif t.shape[0] >= n_eeg:
                padded = np.zeros((n_eeg + n_eog, t.shape[1]), dtype=t.dtype)
                padded[: t.shape[0], :] = t
                data_list.append(padded)
            else:
                data_list.append(t)

        min_samples = min(d.shape[1] for d in data_list)
        data_arr = np.stack([d[:, :min_samples] for d in data_list], axis=0)

        sos = butter(5, [1.0, 45.0], btype="band", fs=_FS, output="sos")
        data_arr[:, :n_eeg, :] = sosfiltfilt(sos, data_arr[:, :n_eeg, :], axis=-1)

        labels_raw = mat.get("labels", mat.get("label", None))
        if labels_raw is None:
            raise EmoKitDataError(f"No label variable found in {mat_path}")
        labels = np.asarray(labels_raw, dtype=np.int64).ravel()

        return data_arr, labels

    def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
        """Load SEED-V data for one subject across selected sessions.

        Args:
            subject_id: 1-based subject identifier (1-16).

        Returns:
            Dict with modality arrays and a ``'labels'`` key.
        """
        all_eeg: list[np.ndarray] = []
        all_eog: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for sess in self.sessions:
            mat_path = self._find_mat(subject_id, sess)
            logger.info("Loading %s", mat_path)

            if self.use_de_features:
                data, labels = self._load_de_features(mat_path)
                all_eeg.append(data)
            else:
                data, labels = self._load_raw_eeg(mat_path)
                all_eeg.append(data[:, :62, :])
                if data.shape[1] > 62:
                    all_eog.append(data[:, 62:65, :])

            all_labels.append(labels)

        result: dict[str, np.ndarray] = {}

        if all_eeg:
            if self.use_de_features:
                result["eeg"] = np.concatenate(all_eeg, axis=0)
            else:
                min_t = min(a.shape[2] for a in all_eeg)
                result["eeg"] = np.concatenate(
                    [a[:, :, :min_t] for a in all_eeg],
                    axis=0,
                )

        if all_eog:
            min_t = min(a.shape[2] for a in all_eog)
            result["eog"] = np.concatenate(
                [a[:, :, :min_t] for a in all_eog],
                axis=0,
            )

        result["labels"] = np.concatenate(all_labels, axis=0)
        return result
