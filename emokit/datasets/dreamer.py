# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""DREAMER dataset loader (14-channel EEG + 2-channel ECG)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from emokit.datasets.base import BaseDataset, _REGISTRY
from emokit.utils import EmoKitDataError

logger = logging.getLogger(__name__)

_EEG_CHANNELS: list[str] = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

_ECG_CHANNELS: list[str] = ["ECG1", "ECG2"]

_N_SUBJECTS: int = 23
_N_VIDEOS: int = 18
_FS: float = 128.0


@_REGISTRY.register("DREAMER")
class DREAMERDataset(BaseDataset):
    """DREAMER: Database for Emotion Recognition through EEG and ECG.

    Reference:
        Katsigiannis & Ramzan, *IEEE J. Biomed. Health Informatics*, 2018.

    The dataset is a single ``DREAMER.mat`` file with nested MATLAB structs.

    Args:
        root: Path to the directory containing ``DREAMER.mat``.
        subjects: Subject IDs to load (1-based, 1–23).
        window_sec: Sliding-window length in seconds.
        overlap: Fractional overlap for the sliding window.
        modalities: Subset of ``{'eeg', 'ecg'}``.
        label_axis: ``'valence'`` or ``'arousal'``.
        label_threshold: Binarisation threshold (default 3.0, scale is 1–5).
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
    ) -> None:
        super().__init__(
            root=root,
            subjects=subjects,
            window_sec=window_sec,
            overlap=overlap,
            modalities=modalities,
        )
        if label_axis not in ("valence", "arousal"):
            raise ValueError(
                f"label_axis must be 'valence' or 'arousal', got '{label_axis}'"
            )
        self.label_axis = label_axis
        self.label_threshold = label_threshold

    def _get_fs(self) -> float:
        return _FS

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
        mat_path = self.root / "DREAMER.mat"
        if not mat_path.exists():
            raise EmoKitDataError(f"DREAMER.mat not found at {mat_path}")
        logger.info("Loading %s", mat_path)
        return loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)

    def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
        """Load DREAMER data for one subject.

        Args:
            subject_id: 1-based subject identifier (1-23).

        Returns:
            Dict with modality arrays and a ``'labels'`` key.
        """
        mat = self._load_mat()

        try:
            dreamer_data = mat["DREAMER"].Data
        except (KeyError, AttributeError) as exc:
            raise EmoKitDataError(
                f"Unexpected DREAMER.mat structure: {exc}"
            ) from exc

        if subject_id < 1 or subject_id > len(dreamer_data):
            raise EmoKitDataError(
                f"Subject {subject_id} out of range [1, {len(dreamer_data)}]"
            )

        subj = dreamer_data[subject_id - 1]

        eeg_trials: list[np.ndarray] = []
        ecg_trials: list[np.ndarray] = []
        labels_list: list[int] = []

        for vid_idx in range(_N_VIDEOS):
            try:
                eeg_stim = np.asarray(
                    subj.EEG.stimuli[vid_idx], dtype=np.float64,
                )
                ecg_stim = np.asarray(
                    subj.ECG.stimuli[vid_idx], dtype=np.float64,
                )
            except (AttributeError, IndexError) as exc:
                logger.warning(
                    "Skipping video %d for subject %d: %s",
                    vid_idx + 1, subject_id, exc,
                )
                continue

            if eeg_stim.ndim == 1:
                eeg_stim = eeg_stim[np.newaxis, :]
            if eeg_stim.ndim == 2 and eeg_stim.shape[1] == 14:
                eeg_stim = eeg_stim.T

            if ecg_stim.ndim == 1:
                ecg_stim = ecg_stim[np.newaxis, :]
            if ecg_stim.ndim == 2 and ecg_stim.shape[1] == 2:
                ecg_stim = ecg_stim.T

            eeg_trials.append(eeg_stim)
            ecg_trials.append(ecg_stim)

            if self.label_axis == "valence":
                rating = float(
                    np.asarray(subj.ScoreValence).ravel()[vid_idx]
                )
            else:
                rating = float(
                    np.asarray(subj.ScoreArousal).ravel()[vid_idx]
                )
            labels_list.append(
                1 if rating >= self.label_threshold else 0
            )

        if not eeg_trials:
            raise EmoKitDataError(
                f"No valid trials for subject {subject_id}"
            )

        min_t = min(e.shape[1] for e in eeg_trials)
        eeg_arr = np.stack([e[:, :min_t] for e in eeg_trials], axis=0)

        result: dict[str, np.ndarray] = {
            "eeg": eeg_arr,
            "labels": np.asarray(labels_list, dtype=np.int64),
        }

        if ecg_trials:
            min_t_ecg = min(e.shape[1] for e in ecg_trials)
            result["ecg"] = np.stack(
                [e[:, :min_t_ecg] for e in ecg_trials], axis=0,
            )

        return result
