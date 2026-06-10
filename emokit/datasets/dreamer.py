# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""DREAMER dataset loader (14-channel EEG + 2-channel ECG).

File structure (confirmed)::

    DREAMER.mat  →  loadmat(squeeze_me=True, struct_as_record=False)
    dreamer.Data[i]           # subject i (0-indexed, 23 subjects)
    subject.EEG.stimuli[k]    # video k EEG, shape (M, 14) at 128 Hz
    subject.EEG.baseline[k]   # baseline EEG, shape (M_base, 14)
    subject.ECG.stimuli[k]    # video k ECG, shape (M, 2) at 256 Hz
    subject.ECG.baseline[k]   # baseline ECG, shape (M_base, 2)
    subject.ScoreValence[k]   # 1–5 float
    subject.ScoreArousal[k]   # 1–5 float
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from emokit.datasets.base import _REGISTRY, BaseDataset, segment_trials
from emokit.utils import EmoKitDataError, get_data_root

logger = logging.getLogger(__name__)

_EEG_CHANNELS: list[str] = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]

_ECG_CHANNELS: list[str] = ["ECG1", "ECG2"]

_N_SUBJECTS: int = 23
_N_VIDEOS: int = 18
_EEG_FS: float = 128.0
_ECG_FS: float = 256.0


@_REGISTRY.register("DREAMER")
class DREAMERDataset(BaseDataset):
    """DREAMER: Database for Emotion Recognition through EEG and ECG.

    Reference:
        Katsigiannis & Ramzan, *IEEE J. Biomed. Health Informatics*, 2018.

    The dataset is a single ``DREAMER.mat`` file with nested MATLAB structs.
    EEG stimuli arrays have shape ``(M, 14)`` (samples × channels) at 128 Hz.
    ECG stimuli arrays have shape ``(M, 2)`` at 256 Hz (downsampled to 128 Hz
    by default).

    Args:
        root: Directory containing ``DREAMER.mat``.  Defaults to
            ``$EMOKIT_DATA_ROOT/DREAMER`` (see ``get_data_root()``).
        subjects: Subject IDs to load (1-based, 1–23).
        window_sec: Sliding-window length in seconds.
        overlap: Fractional overlap for the sliding window.
        modalities: Subset of ``{'eeg', 'ecg'}``.
        label_axis: ``'valence'`` or ``'arousal'``.
        label_threshold: Binarisation threshold (default 3.0, scale 1–5).
            Labels are ``> threshold`` → 1, ``<= threshold`` → 0.
        baseline_correct: If ``True``, subtract per-channel baseline mean
            from each trial before windowing.
    """

    def __init__(
        self,
        root: str | None = None,
        subjects: list[int] | None = None,
        window_sec: float = 4.0,
        overlap: float = 0.5,
        modalities: list[str] | None = None,
        label_axis: str = "valence",
        label_threshold: float = 3.0,
        baseline_correct: bool = True,
    ) -> None:
        if root is None:
            root = str(get_data_root() / "DREAMER")
        super().__init__(
            root=root,
            subjects=subjects,
            window_sec=window_sec,
            overlap=overlap,
            modalities=modalities,
        )
        if label_axis not in ("valence", "arousal"):
            raise ValueError(
                f"label_axis must be 'valence' or 'arousal', " f"got '{label_axis}'"
            )
        self.label_axis = label_axis
        self.label_threshold = label_threshold
        self.baseline_correct = baseline_correct
        self._mat_cache: dict | None = None

    def _get_fs(self) -> float:
        return _EEG_FS

    def is_pre_extracted(self) -> bool:
        """DREAMER provides raw signals, not pre-extracted features."""
        return False

    def get_subject_ids(self) -> list[int]:
        return list(range(1, _N_SUBJECTS + 1))

    def get_channel_names(self, modality: str) -> list[str]:
        if modality == "eeg":
            return list(_EEG_CHANNELS)
        if modality == "ecg":
            return list(_ECG_CHANNELS)
        raise EmoKitDataError(f"Unknown DREAMER modality '{modality}'")

    def get_label_names(self) -> list[str]:
        return ["low", "high"]

    def _load_mat(self) -> dict:
        """Load and cache the ``DREAMER.mat`` file."""
        if self._mat_cache is not None:
            return self._mat_cache
        mat_path = self.root / "DREAMER.mat"
        if not mat_path.exists():
            raise EmoKitDataError(f"DREAMER.mat not found at {mat_path}")
        logger.info("Loading %s", mat_path)
        self._mat_cache = loadmat(
            str(mat_path), squeeze_me=True, struct_as_record=False
        )
        return self._mat_cache

    def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
        """Load windowed and baseline-corrected data for one subject.

        Each video's EEG ``(M, 14)`` is transposed to ``(14, M)``,
        optionally baseline-corrected, then sliced into sliding windows
        via :func:`segment_trials`.  Windows from all 18 videos are
        concatenated, and each window inherits its video's binary label.

        Args:
            subject_id: 1-based identifier (1–23).

        Returns:
            Dict with ``'eeg'`` of shape ``(N_windows, 14, win_samples)``,
            optionally ``'ecg'`` of shape ``(N_windows, 2, win_samples)``,
            and ``'labels'`` of shape ``(N_windows,)``.
        """
        mat = self._load_mat()

        try:
            dreamer_data = mat["DREAMER"].Data
        except (KeyError, AttributeError) as exc:
            raise EmoKitDataError(f"Unexpected DREAMER.mat structure: {exc}") from exc

        if subject_id < 1 or subject_id > len(dreamer_data):
            raise EmoKitDataError(
                f"Subject {subject_id} out of range " f"[1, {len(dreamer_data)}]"
            )

        subj = dreamer_data[subject_id - 1]

        win_samples = int(round(self.window_sec * _EEG_FS))

        all_eeg_windows: list[np.ndarray] = []
        all_ecg_windows: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        modalities = self.modalities or ["eeg"]
        want_ecg = "ecg" in modalities

        for vid_idx in range(_N_VIDEOS):
            # --- EEG ---
            try:
                eeg_raw = np.asarray(subj.EEG.stimuli[vid_idx], dtype=np.float64)
            except (AttributeError, IndexError) as exc:
                logger.warning(
                    "Skipping video %d for subject %d: %s",
                    vid_idx + 1,
                    subject_id,
                    exc,
                )
                continue

            # (M, 14) → (14, M)
            if eeg_raw.ndim == 2 and eeg_raw.shape[1] == 14:
                eeg = eeg_raw.T
            elif eeg_raw.ndim == 2 and eeg_raw.shape[0] == 14:
                eeg = eeg_raw
            else:
                logger.warning(
                    "Unexpected EEG shape %s for subject %d, " "video %d; skipping",
                    eeg_raw.shape,
                    subject_id,
                    vid_idx + 1,
                )
                continue

            # Baseline correction
            if self.baseline_correct:
                try:
                    bl_raw = np.asarray(
                        subj.EEG.baseline[vid_idx],
                        dtype=np.float64,
                    )
                    if bl_raw.ndim == 2 and bl_raw.shape[1] == 14:
                        bl = bl_raw.T
                    elif bl_raw.ndim == 2 and bl_raw.shape[0] == 14:
                        bl = bl_raw
                    else:
                        bl = bl_raw.reshape(14, -1)
                    eeg = eeg - bl.mean(axis=1, keepdims=True)
                except Exception:
                    logger.debug(
                        "Baseline correction failed for subject %d, "
                        "video %d; using raw EEG",
                        subject_id,
                        vid_idx + 1,
                    )

            # Skip if trial is shorter than one window
            if eeg.shape[1] < win_samples:
                logger.debug(
                    "Video %d too short (%d < %d); skipping",
                    vid_idx + 1,
                    eeg.shape[1],
                    win_samples,
                )
                continue

            # Window the EEG: (1, 14, M) → (n_win, 14, win_samples)
            eeg_3d = eeg[np.newaxis, :, :]
            eeg_wins = segment_trials(eeg_3d, _EEG_FS, self.window_sec, self.overlap)
            n_win = eeg_wins.shape[0]
            all_eeg_windows.append(eeg_wins)

            # --- ECG (optional) ---
            if want_ecg:
                try:
                    ecg_raw = np.asarray(
                        subj.ECG.stimuli[vid_idx],
                        dtype=np.float64,
                    )
                    if ecg_raw.ndim == 2 and ecg_raw.shape[1] == 2:
                        ecg = ecg_raw.T  # (2, M_ecg)
                    elif ecg_raw.ndim == 2 and ecg_raw.shape[0] == 2:
                        ecg = ecg_raw
                    else:
                        ecg = ecg_raw.reshape(2, -1)

                    # Downsample 256 Hz → 128 Hz
                    ecg = resample_poly(ecg, up=1, down=2, axis=1)

                    if ecg.shape[1] >= win_samples:
                        ecg_3d = ecg[np.newaxis, :, :]
                        ecg_wins = segment_trials(
                            ecg_3d,
                            _EEG_FS,
                            self.window_sec,
                            self.overlap,
                        )
                        # Match EEG window count
                        ecg_wins = ecg_wins[:n_win]
                        if ecg_wins.shape[0] == n_win:
                            all_ecg_windows.append(ecg_wins)
                except Exception:
                    logger.debug(
                        "ECG processing failed for subject %d, " "video %d",
                        subject_id,
                        vid_idx + 1,
                    )

            # --- Label ---
            if self.label_axis == "valence":
                rating = float(np.asarray(subj.ScoreValence).ravel()[vid_idx])
            else:
                rating = float(np.asarray(subj.ScoreArousal).ravel()[vid_idx])
            label = 1 if rating > self.label_threshold else 0
            all_labels.append(np.full(n_win, label, dtype=np.int64))

        if not all_eeg_windows:
            raise EmoKitDataError(f"No valid EEG windows for subject {subject_id}")

        result: dict[str, np.ndarray] = {
            "eeg": np.concatenate(all_eeg_windows, axis=0),
            "labels": np.concatenate(all_labels, axis=0),
        }

        if all_ecg_windows and len(all_ecg_windows) == len(all_eeg_windows):
            result["ecg"] = np.concatenate(all_ecg_windows, axis=0)

        logger.info(
            "Subject %d: %d EEG windows, labels %s",
            subject_id,
            result["eeg"].shape[0],
            dict(zip(*np.unique(result["labels"], return_counts=True))),
        )

        return result
