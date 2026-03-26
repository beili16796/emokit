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
_N_TRIALS_PER_SESSION: int = 15
_FS: float = 200.0
_N_EEG_CH: int = 62
_N_BANDS: int = 5

# Labels in label.mat use {-1, 0, 1} → remap to {0, 1, 2}
_LABEL_REMAP: dict[int, int] = {-1: 0, 0: 1, 1: 2}


@_REGISTRY.register("SEED")
class SEEDDataset(BaseDataset):
    """SEED: SJTU Emotion EEG Dataset (three emotions).

    Reference:
        Zheng & Lu, *IEEE Trans. Autonomous Mental Development*, 2015.

    Supports loading from the ``ExtractedFeatures/`` directory where each
    ``.mat`` file contains 15 trial variables named ``{prefix}_de_LDS{1..15}``
    with shape ``(62, n_windows, 5)``.

    Subject-to-file mapping is inferred from the directory listing:
    files are sorted alphabetically and grouped into 15 subjects × 3 sessions.

    Args:
        root: Path to the SEED data directory.
        subjects: Subject IDs to load (1-based, 1–15).
        window_sec: Sliding-window length in seconds (unused for pre-extracted).
        overlap: Fractional overlap (unused for pre-extracted).
        modalities: Must be ``['eeg']`` or ``None``.
        sessions: Which sessions to load (1-based); ``None`` loads all.
        use_de_features: If ``True`` (default), load DE features from
            ``ExtractedFeatures/``.
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
        self._subject_file_map: dict[int, list[Path]] | None = None

    @property
    def is_pre_extracted(self) -> bool:
        return self.use_de_features

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

    # ------------------------------------------------------------------
    # Subject-to-file mapping
    # ------------------------------------------------------------------

    def _build_subject_file_map(self) -> dict[int, list[Path]]:
        """Map subject_id (1-based) to a list of 3 session .mat files.

        Files in ``ExtractedFeatures/`` are named like
        ``{name}_{date}.mat``.  We sort them alphabetically and group
        every 3 consecutive files as one subject's 3 sessions.
        """
        ef_dir = self.root / "ExtractedFeatures"
        if not ef_dir.is_dir():
            ef_dir = self.root

        mat_files = sorted(p for p in ef_dir.glob("*.mat") if p.name != "label.mat")

        if len(mat_files) < _N_SUBJECTS * _N_SESSIONS:
            # Try grouping by prefix (subject name before date)
            from collections import defaultdict

            groups: dict[str, list[Path]] = defaultdict(list)
            for p in mat_files:
                prefix = p.stem.rsplit("_", 1)[0]
                groups[prefix].append(p)
            mat_files = []
            for prefix in sorted(groups):
                mat_files.extend(sorted(groups[prefix]))

        mapping: dict[int, list[Path]] = {}
        for sid in range(1, _N_SUBJECTS + 1):
            start = (sid - 1) * _N_SESSIONS
            end = start + _N_SESSIONS
            if end <= len(mat_files):
                mapping[sid] = mat_files[start:end]
            else:
                logger.warning("Not enough files for subject %d", sid)
        return mapping

    def _get_subject_files(self, subject_id: int) -> list[Path]:
        if self._subject_file_map is None:
            self._subject_file_map = self._build_subject_file_map()
        files = self._subject_file_map.get(subject_id, [])
        if not files:
            raise EmoKitDataError(
                f"No files found for SEED subject {subject_id} under {self.root}"
            )
        return files

    # ------------------------------------------------------------------
    # Label loading
    # ------------------------------------------------------------------

    def _load_labels(self) -> np.ndarray:
        """Load the 15-trial label vector and remap {-1,0,1} → {0,1,2}."""
        label_paths = [
            self.root / "Preprocessed_EEG" / "label.mat",
            self.root / "ExtractedFeatures" / "label.mat",
            self.root / "label.mat",
        ]
        label_path = next((p for p in label_paths if p.exists()), None)
        if label_path is None:
            raise EmoKitDataError(
                f"label.mat not found under {self.root}. "
                f"Tried: {[str(p) for p in label_paths]}"
            )
        mat = loadmat(str(label_path), squeeze_me=True)
        raw = np.asarray(mat["label"]).flatten().astype(int)
        return np.array([_LABEL_REMAP.get(v, v) for v in raw], dtype=np.int64)

    # ------------------------------------------------------------------
    # DE feature loading
    # ------------------------------------------------------------------

    def _load_de_from_mat(self, mat_path: Path) -> list[np.ndarray]:
        """Extract per-trial DE arrays from a session .mat file.

        Variables matching ``*_de_LDS*`` are collected. Each has shape
        ``(62, n_windows, 5)`` and is transposed to ``(n_windows, 62, 5)``.

        Returns:
            List of 15 arrays, one per trial.
        """
        mat = loadmat(str(mat_path), squeeze_me=False)

        de_keys = sorted(
            (k for k in mat if "de_LDS" in k and not k.startswith("__")),
            key=lambda k: int(
                "".join(c for c in k.split("de_LDS")[-1] if c.isdigit()) or "0"
            ),
        )

        if not de_keys:
            all_keys = [k for k in mat if not k.startswith("__")]
            raise EmoKitDataError(
                f"No *de_LDS* variables in {mat_path}. "
                f"Available keys ({len(all_keys)}): {all_keys[:20]}"
            )

        trials: list[np.ndarray] = []
        for k in de_keys:
            arr = np.asarray(mat[k], dtype=np.float64)
            if arr.ndim == 3 and arr.shape[0] == _N_EEG_CH:
                # (62, n_windows, 5) → (n_windows, 62, 5)
                arr = np.transpose(arr, (1, 0, 2))
            elif arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            trials.append(arr)

        return trials

    def _load_raw_eeg(self, mat_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Load raw EEG signals from a Preprocessed_EEG ``.mat`` file."""
        mat = loadmat(str(mat_path), squeeze_me=True)

        data_keys = [
            k for k in mat if not k.startswith("__") and k not in ("labels", "label")
        ]
        trials: list[np.ndarray] = []
        for key in sorted(data_keys):
            arr = np.asarray(mat[key], dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] >= _N_EEG_CH:
                trials.append(arr[:_N_EEG_CH, :])

        if not trials:
            raise EmoKitDataError(f"No trial data found in {mat_path}")

        min_samples = min(t.shape[1] for t in trials)
        data = np.stack([t[:, :min_samples] for t in trials], axis=0)

        labels_raw = mat.get("labels", mat.get("label", None))
        if labels_raw is None:
            raise EmoKitDataError(f"No label variable found in {mat_path}")
        labels = np.asarray(labels_raw, dtype=np.int64).ravel()

        return data, labels

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
        """Load SEED data for one subject across selected sessions.

        For pre-extracted DE: loads from ``ExtractedFeatures/*.mat``,
        each containing 15 ``*_de_LDS*`` variables per session.
        """
        session_labels = self._load_labels()  # (15,) per session

        if self.use_de_features:
            files = self._get_subject_files(subject_id)
            all_de: list[np.ndarray] = []
            all_labels: list[np.ndarray] = []

            for sess_idx, sess in enumerate(self.sessions):
                if sess_idx >= len(files):
                    break
                mat_path = files[sess_idx]
                logger.info("Loading DE from %s", mat_path)

                trial_des = self._load_de_from_mat(mat_path)
                for trial_idx, de in enumerate(trial_des):
                    if trial_idx < len(session_labels):
                        lbl = session_labels[trial_idx]
                        all_de.append(de)
                        all_labels.append(np.full(de.shape[0], lbl, dtype=np.int64))

            if not all_de:
                raise EmoKitDataError(
                    f"No DE data loaded for SEED subject {subject_id}"
                )

            eeg = np.concatenate(all_de, axis=0)
            labels = np.concatenate(all_labels, axis=0)
            return {"eeg": eeg, "labels": labels}

        # Raw EEG fallback
        all_data: list[np.ndarray] = []
        all_labels_raw: list[np.ndarray] = []

        for sess in self.sessions:
            files = self._get_subject_files(subject_id)
            if sess - 1 >= len(files):
                break
            mat_path = files[sess - 1]
            logger.info("Loading raw EEG from %s", mat_path)
            data, labels = self._load_raw_eeg(mat_path)
            all_data.append(data)
            all_labels_raw.append(labels)

        if not all_data:
            raise EmoKitDataError(f"No raw EEG loaded for SEED subject {subject_id}")

        min_t = min(a.shape[2] for a in all_data)
        eeg = np.concatenate([a[:, :, :min_t] for a in all_data], axis=0)

        return {
            "eeg": eeg,
            "labels": np.concatenate(all_labels_raw, axis=0),
        }
