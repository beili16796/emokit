# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""SEED-V dataset loader (62-channel EEG + 3-channel EOG, 5 emotions)."""

from __future__ import annotations

import logging
import pickle
import warnings
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
_EYE_FEATURE_DIM: int = 33
_EYE_FEATURE_NAMES: list[str] = [f"eye_feat_{i:02d}" for i in range(_EYE_FEATURE_DIM)]

_EMOTION_LABELS: list[str] = ["happy", "fear", "neutral", "sad", "disgust"]

_N_SUBJECTS: int = 16
_N_SESSIONS: int = 3
_N_TRIALS_PER_SESSION: int = 15
_N_TOTAL_TRIALS: int = _N_SESSIONS * _N_TRIALS_PER_SESSION  # 45
_FS: float = 200.0
_N_EEG_CH: int = 62
_N_BANDS: int = 5

SEED_62_CHANNELS: list[str] = list(_EEG_CHANNELS)


@_REGISTRY.register("SEED-V")
class SEEDVDataset(BaseDataset):
    """SEED-V: SJTU Emotion EEG Dataset (five emotions).

    Reference:
        Liu et al., *IEEE Trans. Affective Computing*, 2021.

    Supports two on-disk formats:

    1. **NPZ** (preferred): ``EEG_DE_features/{subject_id}_123.npz`` containing
       pickle-serialized dicts of pre-extracted DE features with shape
       ``(n_windows, 310)`` per trial (310 = 62 channels × 5 bands).
    2. **MAT**: Per-session ``.mat`` files with ``de_LDS`` variable.

    Args:
        root: Path to the SEED-V data directory.
        subjects: Subject IDs to load (1-based, 1–16).
        window_sec: Sliding-window length in seconds (unused for pre-extracted).
        overlap: Fractional overlap (unused for pre-extracted).
        modalities: Subset of ``{'eeg', 'eog'}`` to include.
        sessions: Which sessions to load (1-based); ``None`` loads all.
        use_de_features: If ``True`` (default), load pre-extracted DE features.
    """

    def __init__(
        self,
        root: str | None = None,
        subjects: list[int] | None = None,
        window_sec: float = 4.0,
        overlap: float = 0.5,
        modalities: list[str] | None = None,
        sessions: list[int] | None = None,
        use_de_features: bool = True,
        use_eye_features: bool = True,
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
        self.use_eye_features = use_eye_features
        self._pre_extracted: bool = use_de_features

    @property
    def is_pre_extracted(self) -> bool:
        return self._pre_extracted

    def _get_fs(self) -> float:
        return _FS

    def get_subject_ids(self) -> list[int]:
        return list(range(1, _N_SUBJECTS + 1))

    def get_channel_names(self, modality: str) -> list[str]:
        if modality == "eeg":
            return list(_EEG_CHANNELS)
        if modality == "eog":
            return (
                list(_EYE_FEATURE_NAMES)
                if self.use_de_features and self.use_eye_features
                else list(_EOG_CHANNELS)
            )
        raise EmoKitDataError(f"Unknown SEED-V modality '{modality}'")

    def get_label_names(self) -> list[str]:
        return list(_EMOTION_LABELS)

    # ------------------------------------------------------------------
    # NPZ loading (primary format on local disk)
    # ------------------------------------------------------------------

    def _find_npz(self, subject_id: int) -> Path | None:
        """Find the NPZ file for a subject."""
        patterns = [
            self.root / "EEG_DE_features" / f"{subject_id}_123.npz",
            self.root / f"{subject_id}_123.npz",
        ]
        for p in patterns:
            if p.exists():
                return p
        return None

    def _find_eye_npz(self, subject_id: int) -> Path | None:
        """Find the eye-feature NPZ file for a subject."""
        patterns = [
            self.root / "Eye_movement_features" / f"{subject_id}_123.npz",
        ]
        for p in patterns:
            if p.exists():
                return p
        return None

    def _load_npz(self, npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load pre-extracted DE features from NPZ (pickle-encoded dicts).

        The NPZ contains:
        - ``data``: pickle-serialized dict mapping trial_id (int) to
          array of shape ``(n_windows, 310)`` where 310 = 62ch × 5bands.
        - ``label``: pickle-serialized dict mapping trial_id to
          per-window label arrays.

        Returns:
            de: ``(total_windows, 62, 5)``
            labels: ``(total_windows,)``
        """
        npz = np.load(str(npz_path), allow_pickle=True)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*dtype.*align.*",
                category=np.VisibleDeprecationWarning,
            )
            data_raw = npz["data"]
            if data_raw.shape == ():
                data_dict = pickle.loads(data_raw.item())
            elif hasattr(data_raw, "item"):
                data_dict = data_raw.item()
            else:
                data_dict = data_raw

            label_raw = npz["label"]
            if label_raw.shape == ():
                label_dict = pickle.loads(label_raw.item())
            elif hasattr(label_raw, "item"):
                label_dict = label_raw.item()
            else:
                label_dict = label_raw

        de_list: list[np.ndarray] = []
        lbl_list: list[np.ndarray] = []

        for trial_id in range(_N_TOTAL_TRIALS):
            trial_de = np.asarray(data_dict[trial_id], dtype=np.float64)
            n_windows = trial_de.shape[0]
            de_reshaped = trial_de.reshape(n_windows, _N_EEG_CH, _N_BANDS)
            de_list.append(de_reshaped)

            trial_lbl = np.asarray(label_dict[trial_id], dtype=np.int64)
            if trial_lbl.ndim == 0:
                trial_lbl = np.full(n_windows, int(trial_lbl), dtype=np.int64)
            elif len(trial_lbl) != n_windows:
                trial_lbl = np.full(n_windows, int(trial_lbl[0]), dtype=np.int64)
            lbl_list.append(trial_lbl)

        de = np.concatenate(de_list, axis=0)
        labels = np.concatenate(lbl_list, axis=0)
        return de, labels

    def _load_eye_npz(self, npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load pre-extracted eye-movement features from NPZ."""
        npz = np.load(str(npz_path), allow_pickle=True)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*dtype.*align.*",
                category=np.VisibleDeprecationWarning,
            )
            data_raw = npz["data"]
            if data_raw.shape == ():
                data_dict = pickle.loads(data_raw.item())
            elif hasattr(data_raw, "item"):
                data_dict = data_raw.item()
            else:
                data_dict = data_raw

            label_raw = npz["label"]
            if label_raw.shape == ():
                label_dict = pickle.loads(label_raw.item())
            elif hasattr(label_raw, "item"):
                label_dict = label_raw.item()
            else:
                label_dict = label_raw

        feat_list: list[np.ndarray] = []
        lbl_list: list[np.ndarray] = []
        for trial_id in range(_N_TOTAL_TRIALS):
            trial_feat = np.asarray(data_dict[trial_id], dtype=np.float64)
            if trial_feat.ndim == 1:
                trial_feat = trial_feat[:, None]
            feat_list.append(trial_feat)

            trial_lbl = np.asarray(label_dict[trial_id], dtype=np.int64)
            n_windows = trial_feat.shape[0]
            if trial_lbl.ndim == 0:
                trial_lbl = np.full(n_windows, int(trial_lbl), dtype=np.int64)
            elif len(trial_lbl) != n_windows:
                trial_lbl = np.full(n_windows, int(trial_lbl[0]), dtype=np.int64)
            lbl_list.append(trial_lbl)

        feats = np.concatenate(feat_list, axis=0)
        labels = np.concatenate(lbl_list, axis=0)
        return feats, labels

    # ------------------------------------------------------------------
    # MAT loading (fallback)
    # ------------------------------------------------------------------

    def _find_mat(self, subject_id: int, session: int) -> Path:
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

    def _load_de_features_mat(self, mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load pre-computed DE features from a ``.mat`` file."""
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
                    trials.append(cell.reshape(-1, _N_EEG_CH, _N_BANDS))
            de = np.concatenate(trials, axis=0)
        else:
            de = np.asarray(raw_de, dtype=np.float64)
            if de.ndim == 2:
                de = de[np.newaxis, ...]
            elif de.ndim == 3 and de.shape[0] == _N_EEG_CH:
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
                    "Label count (%d) != DE window count (%d) " "in %s; truncating",
                    len(labels),
                    de.shape[0],
                    mat_path,
                )
                n = min(len(labels), de.shape[0])
                de, labels = de[:n], labels[:n]

        return de, labels

    def _load_raw_eeg(self, mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load raw EEG signals from a ``.mat`` file and band-pass filter."""
        mat = loadmat(str(mat_path), squeeze_me=True)

        data_keys = [
            k for k in mat if not k.startswith("__") and k not in ("labels", "label")
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

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
        """Load SEED-V data for one subject.

        Tries NPZ first (``EEG_DE_features/{id}_123.npz``), then falls
        back to per-session ``.mat`` files.
        """
        npz_path = self._find_npz(subject_id)
        if npz_path is not None and self.use_de_features:
            logger.info("Loading %s via NPZ", npz_path)
            de, labels = self._load_npz(npz_path)
            self._pre_extracted = True
            result: dict[str, np.ndarray] = {"eeg": de, "labels": labels}
            eye_npz_path = self._find_eye_npz(subject_id)
            if eye_npz_path is not None and self.use_eye_features:
                logger.info("Loading %s via NPZ", eye_npz_path)
                eye_feat, eye_labels = self._load_eye_npz(eye_npz_path)
                if len(eye_labels) != len(labels) or not np.array_equal(
                    eye_labels, labels
                ):
                    logger.warning(
                        "Eye feature labels mismatch for subject %d; using EEG labels",
                        subject_id,
                    )
                result["eog"] = eye_feat
            return result

        all_eeg: list[np.ndarray] = []
        all_eog: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for sess in self.sessions:
            mat_path = self._find_mat(subject_id, sess)
            logger.info("Loading %s", mat_path)

            if self.use_de_features:
                data, labels = self._load_de_features_mat(mat_path)
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
