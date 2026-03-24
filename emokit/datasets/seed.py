# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""SEED dataset loader (62-channel EEG, 3 emotions)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.io import loadmat

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

_EMOTION_LABELS: list[str] = ["negative", "neutral", "positive"]

_N_SUBJECTS: int = 15
_N_SESSIONS: int = 3
_FS: float = 200.0


@_REGISTRY.register("SEED")
class SEEDDataset(BaseDataset):
    """SEED: SJTU Emotion EEG Dataset (three emotions).

    Reference:
        Zheng & Lu, *IEEE Trans. Autonomous Mental Development*, 2015.

    The dataset is distributed as ``.mat`` files with either raw signals
    or pre-computed differential entropy (DE) features.

    Args:
        root: Path to the SEED data directory.
        subjects: Subject IDs to load (1-based, 1–15).
        window_sec: Sliding-window length in seconds.
        overlap: Fractional overlap for the sliding window.
        modalities: Must be ``['eeg']`` or ``None``.
        sessions: Which sessions to load (1-based); ``None`` loads all.
        use_de_features: If ``True``, load DE features directly from mat.
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

    def _get_fs(self) -> float:
        return _FS

    def get_subject_ids(self) -> list[int]:
        return list(range(1, _N_SUBJECTS + 1))

    def get_channel_names(self, modality: str) -> list[str]:
        if modality == "eeg":
            return list(_EEG_CHANNELS)
        raise EmoKitDataError(f"Unknown SEED modality '{modality}'")

    def get_label_names(self) -> list[str]:
        return list(_EMOTION_LABELS)

    def _find_mat(self, subject_id: int, session: int) -> Path:
        """Resolve the ``.mat`` path for a given subject/session pair.

        Searches several common SEED directory layouts including the
        ``Preprocessed_EEG/`` convention with ``{subject}_{session}_de_LDS.mat``.
        """
        patterns = [
            self.root / f"{subject_id}" / f"{session}.mat",
            self.root / f"sub{subject_id}" / f"session{session}.mat",
            self.root / f"s{subject_id:02d}" / f"sess{session:02d}.mat",
            self.root / f"{subject_id}_{session}.mat",
            self.root / "Preprocessed_EEG" / f"{subject_id}_{session}.mat",
            self.root / "Preprocessed_EEG" / f"{subject_id}_{session}_de_LDS.mat",
        ]
        for p in patterns:
            if p.exists():
                return p

        if (self.root / "Preprocessed_EEG").is_dir():
            import glob as _glob
            found = sorted(
                Path(m)
                for m in _glob.glob(str(self.root / "Preprocessed_EEG" / f"*{subject_id}*{session}*.mat"))
            )
            if found:
                return found[0]

        raise EmoKitDataError(
            f"No .mat file for subject {subject_id}, session {session} "
            f"under {self.root}. Tried: {[str(p) for p in patterns]}"
        )

    def _load_de_features(self, mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load pre-computed DE features.

        Handles both numeric arrays and MATLAB cell arrays (``object`` dtype)
        for the ``de_LDS`` variable.

        Returns:
            data: ``(n_windows, n_channels, n_bands)``
            labels: ``(n_windows,)``
        """
        mat = loadmat(str(mat_path), squeeze_me=True)

        de_key = None
        for key in ("de_LDS", "de_lds", "DE_LDS", "de_movingAve", "de_LDS1"):
            if key in mat:
                de_key = key
                break
        if de_key is None:
            raise EmoKitDataError(
                f"No DE feature variable found in {mat_path}. "
                f"Available keys: {[k for k in mat if not k.startswith('__')]}"
            )

        raw_de = mat[de_key]

        if raw_de.dtype == object:
            trials: list[np.ndarray] = []
            for i in range(len(raw_de)):
                cell = np.asarray(raw_de[i], dtype=np.float64)
                if cell.ndim == 2:
                    trials.append(cell[np.newaxis, ...])
                elif cell.ndim == 3:
                    trials.append(np.transpose(cell, (2, 0, 1)))
                else:
                    trials.append(cell.reshape(-1, 62, 5))
            de = np.concatenate(trials, axis=0)
        else:
            de = np.asarray(raw_de, dtype=np.float64)
            if de.ndim == 2:
                de = de[np.newaxis, ...]
            elif de.ndim == 3 and de.shape[0] == 62:
                de = np.transpose(de, (2, 0, 1))

        labels_raw = mat.get("labels", mat.get("label", None))
        if labels_raw is None:
            raise EmoKitDataError(f"No label variable found in {mat_path}")
        labels = np.asarray(labels_raw, dtype=np.int64).ravel()

        if len(labels) != de.shape[0]:
            if len(labels) == 1:
                labels = np.repeat(labels, de.shape[0])
            else:
                logger.warning(
                    "Label count (%d) != DE window count (%d) in %s; truncating",
                    len(labels), de.shape[0], mat_path,
                )
                n = min(len(labels), de.shape[0])
                de, labels = de[:n], labels[:n]

        return de, labels

    def _load_raw_eeg(self, mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load raw EEG signals from a ``.mat`` file.

        Returns:
            data: ``(n_trials, 62, n_samples)``
            labels: ``(n_trials,)``
        """
        mat = loadmat(str(mat_path), squeeze_me=True)

        data_keys = [
            k for k in mat if not k.startswith("__") and k not in ("labels", "label")
        ]
        trials: list[np.ndarray] = []
        for key in sorted(data_keys):
            arr = np.asarray(mat[key], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] >= 62:
                trials.append(arr[:62, :])

        if not trials:
            raise EmoKitDataError(f"No trial data found in {mat_path}")

        min_samples = min(t.shape[1] for t in trials)
        data = np.stack([t[:, :min_samples] for t in trials], axis=0)

        labels_raw = mat.get("labels", mat.get("label", None))
        if labels_raw is None:
            raise EmoKitDataError(f"No label variable found in {mat_path}")
        labels = np.asarray(labels_raw, dtype=np.int64).ravel()

        return data, labels

    def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
        """Load SEED data for one subject across selected sessions.

        Args:
            subject_id: 1-based subject identifier (1-15).

        Returns:
            Dict with ``'eeg'`` and ``'labels'`` keys.
        """
        all_data: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for sess in self.sessions:
            mat_path = self._find_mat(subject_id, sess)
            logger.info("Loading %s", mat_path)

            if self.use_de_features:
                data, labels = self._load_de_features(mat_path)
            else:
                data, labels = self._load_raw_eeg(mat_path)

            all_data.append(data)
            all_labels.append(labels)

        if self.use_de_features:
            eeg = np.concatenate(all_data, axis=0)
        else:
            min_t = min(a.shape[2] for a in all_data)
            eeg = np.concatenate([a[:, :, :min_t] for a in all_data], axis=0)

        return {
            "eeg": eeg,
            "labels": np.concatenate(all_labels, axis=0),
        }
