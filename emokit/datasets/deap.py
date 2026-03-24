# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""DEAP dataset loader (32-channel EEG + peripheral physiology)."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from emokit.datasets.base import _REGISTRY, BaseDataset
from emokit.utils import EmoKitDataError

logger = logging.getLogger(__name__)

_EEG_CHANNELS: list[str] = [
    "Fp1",
    "AF3",
    "F3",
    "F7",
    "FC5",
    "FC1",
    "C3",
    "T7",
    "CP5",
    "CP1",
    "P3",
    "P7",
    "PO3",
    "O1",
    "Oz",
    "Pz",
    "Fp2",
    "AF4",
    "F4",
    "F8",
    "FC6",
    "FC2",
    "C4",
    "T8",
    "CP6",
    "CP2",
    "P4",
    "P8",
    "PO4",
    "O2",
    "Fz",
    "Cz",
]

DEAP_EEG_CHANNELS: list[str] = list(_EEG_CHANNELS)

_PERIPHERAL_CHANNELS: list[str] = [
    "hEOG",
    "vEOG",
    "zEMG",
    "tEMG",
    "GSR",
    "Resp",
    "Temp",
    "Status",
]

_ORIGINAL_FS: float = 512.0
_DOWNSAMPLED_FS: float = 128.0
_N_TRIALS: int = 40
_TRIAL_SEC: float = 60.0
_BASELINE_SEC: float = 3.0


@_REGISTRY.register("DEAP")
class DEAPDataset(BaseDataset):
    """DEAP: Database for Emotion Analysis using Physiological Signals.

    Reference:
        Koelstra et al., *IEEE Trans. Affective Computing*, 2012.

    The dataset ships as either raw ``.bdf`` files or pre-processed ``.dat``
    (pickled) files.  This loader tries ``.bdf`` first and falls back to
    ``.dat``.

    Args:
        root: Path to the DEAP data directory.
        subjects: Subject IDs to load (1-based).
        window_sec: Sliding-window length in seconds.
        overlap: Fractional overlap for the sliding window.
        modalities: Subset of ``{'eeg', 'gsr', 'ecg'}`` to include.
        label_axis: Which label dimension — ``'valence'`` or ``'arousal'``.
        label_threshold: Binarisation threshold (default 5.0).
    """

    _MODALITY_SLICES: dict[str, slice] = {
        "eeg": slice(0, 32),
        "gsr": slice(36, 37),
        "ecg": slice(38, 39),
    }

    def __init__(
        self,
        root: str | None = None,
        subjects: list[int] | None = None,
        window_sec: float = 4.0,
        overlap: float = 0.5,
        modalities: list[str] | None = None,
        label_axis: str = "valence",
        label_threshold: float = 5.0,
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
        return _DOWNSAMPLED_FS

    def get_subject_ids(self) -> list[int]:
        return list(range(1, 33))

    def get_channel_names(self, modality: str) -> list[str]:
        if modality == "eeg":
            return list(_EEG_CHANNELS)
        if modality == "gsr":
            return ["GSR"]
        if modality == "ecg":
            return ["ECG"]
        raise EmoKitDataError(f"Unknown DEAP modality '{modality}'")

    def get_label_names(self) -> list[str]:
        return ["Low", "High"]

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _subject_stem(self, subject_id: int) -> str:
        return f"s{subject_id:02d}"

    def _load_bdf(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Read a ``.bdf`` file via MNE and return ``(data, labels)``.

        Returns:
            data: ``(n_trials, n_channels, n_samples)``
            labels: ``(n_trials,)`` binarised.
        """
        try:
            import mne  # noqa: WPS433 (local import – optional heavy dep)
        except ImportError as exc:
            raise EmoKitDataError(
                "MNE is required for .bdf loading. Install with: pip install mne"
            ) from exc

        raw = mne.io.read_raw_bdf(str(path), preload=True, verbose=False)
        raw.pick_channels(_EEG_CHANNELS, ordered=True)

        raw.filter(
            1.0,
            45.0,
            method="iir",
            iir_params=dict(order=5, ftype="butter"),
            verbose=False,
        )
        raw.set_eeg_reference("average", verbose=False)

        sfreq = raw.info["sfreq"]
        data_arr = raw.get_data()  # (n_channels, total_samples)

        trial_samples = int(_TRIAL_SEC * sfreq)
        baseline_samples = int(_BASELINE_SEC * sfreq)
        trials: list[np.ndarray] = []
        for t in range(_N_TRIALS):
            start = t * (trial_samples + baseline_samples) + baseline_samples
            end = start + trial_samples
            if end > data_arr.shape[1]:
                break
            trials.append(data_arr[:, start:end])

        data = np.stack(trials, axis=0)  # (N, C, T)

        down = int(sfreq / _DOWNSAMPLED_FS)
        if down > 1:
            data = resample_poly(data, up=1, down=down, axis=-1)

        labels_path = path.parent / "labels.npy"
        if labels_path.exists():
            all_labels = np.load(labels_path)
            col = 0 if self.label_axis == "valence" else 1
            labels = (all_labels[:, col] >= self.label_threshold).astype(np.int64)
        else:
            logger.warning("No labels.npy found alongside .bdf – using dummy labels")
            labels = np.zeros(data.shape[0], dtype=np.int64)

        return data, labels

    def _load_dat(self, path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Read a pickled ``.dat`` file (DEAP preprocessed format).

        Returns:
            data: ``(n_trials, n_channels, n_samples)``
            labels: ``(n_trials,)`` binarised.
        """
        with open(path, "rb") as fh:
            content = pickle.load(fh, encoding="latin1")

        data = np.asarray(content["data"], dtype=np.float64)
        raw_labels = np.asarray(content["labels"], dtype=np.float64)

        assert data.ndim == 3, f"Expected 3D array (N,C,T), got {data.shape}"

        baseline_samples = int(_BASELINE_SEC * _DOWNSAMPLED_FS)
        if data.shape[2] > int((_TRIAL_SEC - _BASELINE_SEC) * _DOWNSAMPLED_FS):
            data = data[:, :, baseline_samples:]

        sos = butter(5, [1.0, 45.0], btype="band", fs=_DOWNSAMPLED_FS, output="sos")
        data[:, :32, :] = sosfiltfilt(sos, data[:, :32, :], axis=-1)

        mean = data[:, :32, :].mean(axis=1, keepdims=True)
        data[:, :32, :] -= mean

        col = 0 if self.label_axis == "valence" else 1
        labels = (raw_labels[:, col] >= self.label_threshold).astype(np.int64)

        return data, labels

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
        """Load raw data for one DEAP subject.

        Tries ``.bdf`` first, then ``.dat``.

        Args:
            subject_id: 1-based subject identifier (1-32).

        Returns:
            Dict with modality arrays and a ``'labels'`` key.

        Raises:
            EmoKitDataError: If neither file format is found.
        """
        stem = self._subject_stem(subject_id)

        bdf_path = self.root / f"{stem}.bdf"
        dat_path = self.root / f"{stem}.dat"

        if bdf_path.exists():
            logger.info("Loading %s via BDF", bdf_path)
            data, labels = self._load_bdf(bdf_path)
        elif dat_path.exists():
            logger.info("Loading %s via DAT", dat_path)
            data, labels = self._load_dat(dat_path)
        else:
            raise FileNotFoundError(
                f"DEAP file not found: {dat_path}\n"
                f"Please download from http://eecs.qmul.ac.uk/mmv/datasets/deap/\n"
                f"Set EMOKIT_DATA_ROOT or pass root= explicitly."
            )

        result: dict[str, np.ndarray] = {"labels": labels}
        for mod, slc in self._MODALITY_SLICES.items():
            if slc.stop <= data.shape[1]:
                result[mod] = data[:, slc, :]

        return result

    @staticmethod
    def binarize_labels(
        ratings: np.ndarray,
        threshold: float = 5.0,
    ) -> np.ndarray:
        """Convert continuous V/A ratings to binary labels.

        Args:
            ratings: 1-D array of continuous ratings.
            threshold: Values ``>= threshold`` are mapped to 1.

        Returns:
            Integer array of 0/1 labels.
        """
        return (np.asarray(ratings) >= threshold).astype(np.int64)
