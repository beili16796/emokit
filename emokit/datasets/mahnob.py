# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""MAHNOB-HCI dataset loader (32-channel EEG + ECG + GSR)."""

from __future__ import annotations

import logging
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

_ORIGINAL_FS: float = 256.0
_TARGET_FS: float = 128.0
_N_SUBJECTS: int = 27


@_REGISTRY.register("MAHNOB-HCI")
class MAHNOBHCIDataset(BaseDataset):
    """MAHNOB-HCI: Multimodal Affect-HCI dataset.

    Reference:
        Soleymani et al., *IEEE Trans. Affective Computing*, 2012.

    Data is stored as ``.bdf`` files containing 32-channel EEG, 1-channel
    ECG, and 1-channel GSR.

    Args:
        root: Path to the MAHNOB-HCI data directory.
        subjects: Subject IDs to load (1-based, 1–27).
        window_sec: Sliding-window length in seconds.
        overlap: Fractional overlap for the sliding window.
        modalities: Subset of ``{'eeg', 'ecg', 'gsr'}``.
        label_axis: ``'valence'`` or ``'arousal'``.
        label_threshold: Binarisation threshold (default 5.0).
    """

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
        return _TARGET_FS

    def get_subject_ids(self) -> list[int]:
        return list(range(1, _N_SUBJECTS + 1))

    def get_channel_names(self, modality: str) -> list[str]:
        if modality == "eeg":
            return list(_EEG_CHANNELS)
        if modality == "ecg":
            return ["ECG"]
        if modality == "gsr":
            return ["GSR"]
        raise EmoKitDataError(f"Unknown MAHNOB-HCI modality '{modality}'")

    def get_label_names(self) -> list[str]:
        return ["low", "high"]

    def _find_session_dirs(self, subject_id: int) -> list[Path]:
        """Return all session directories for a subject."""
        subject_dir = self.root / f"Subject{subject_id}"
        if not subject_dir.exists():
            subject_dir = self.root / f"s{subject_id:02d}"
        if not subject_dir.exists():
            raise EmoKitDataError(
                f"Subject directory not found for subject {subject_id} "
                f"under {self.root}"
            )
        sessions = sorted(
            p for p in subject_dir.iterdir() if p.is_dir() or p.suffix == ".bdf"
        )
        return sessions

    def _load_bdf(self, bdf_path: Path) -> tuple[np.ndarray, float]:
        """Read a ``.bdf`` file via MNE.

        Returns:
            data: ``(n_channels, n_samples)``
            sfreq: Original sampling frequency.
        """
        try:
            import mne
        except ImportError as exc:
            raise EmoKitDataError(
                "MNE is required for .bdf loading. Install with: pip install mne"
            ) from exc

        raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose=False)
        return raw.get_data(), raw.info["sfreq"]

    def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
        """Load MAHNOB-HCI data for one subject.

        Args:
            subject_id: 1-based subject identifier (1-27).

        Returns:
            Dict with modality arrays and a ``'labels'`` key.
        """
        session_paths = self._find_session_dirs(subject_id)

        all_eeg: list[np.ndarray] = []
        all_ecg: list[np.ndarray] = []
        all_gsr: list[np.ndarray] = []
        all_labels: list[int] = []

        labels_file = self.root / "labels.npy"
        subject_labels: np.ndarray | None = None
        if labels_file.exists():
            full_labels = np.load(labels_file, allow_pickle=True)
            if isinstance(full_labels, np.ndarray) and full_labels.ndim >= 2:
                subject_labels = full_labels

        for idx, sess_path in enumerate(session_paths):
            bdf_file = sess_path if sess_path.suffix == ".bdf" else None
            if bdf_file is None:
                bdf_candidates = list(sess_path.glob("*.bdf"))
                if not bdf_candidates:
                    logger.warning("No .bdf in %s, skipping", sess_path)
                    continue
                bdf_file = bdf_candidates[0]

            logger.info("Loading %s", bdf_file)
            data, sfreq = self._load_bdf(bdf_file)

            sos = butter(5, [1.0, 45.0], btype="band", fs=sfreq, output="sos")
            eeg = sosfiltfilt(sos, data[:32, :], axis=-1)

            mean = eeg.mean(axis=0, keepdims=True)
            eeg -= mean

            down = int(sfreq / _TARGET_FS) if sfreq > _TARGET_FS else 1
            if down > 1:
                eeg = resample_poly(eeg, up=1, down=down, axis=-1)

            all_eeg.append(eeg)

            if data.shape[0] > 32:
                ecg = data[32:33, :]
                if down > 1:
                    ecg = resample_poly(ecg, up=1, down=down, axis=-1)
                all_ecg.append(ecg)

            if data.shape[0] > 33:
                gsr = data[33:34, :]
                if down > 1:
                    gsr = resample_poly(gsr, up=1, down=down, axis=-1)
                all_gsr.append(gsr)

            if subject_labels is not None and idx < subject_labels.shape[0]:
                col = 0 if self.label_axis == "valence" else 1
                rating = float(subject_labels[idx, col])
                all_labels.append(1 if rating >= self.label_threshold else 0)
            else:
                all_labels.append(0)

        if not all_eeg:
            raise EmoKitDataError(f"No data loaded for subject {subject_id}")

        min_t = min(e.shape[1] for e in all_eeg)
        eeg_arr = np.stack([e[:, :min_t] for e in all_eeg], axis=0)

        result: dict[str, np.ndarray] = {
            "eeg": eeg_arr,
            "labels": np.asarray(all_labels, dtype=np.int64),
        }

        if all_ecg:
            min_t_ecg = min(e.shape[1] for e in all_ecg)
            result["ecg"] = np.stack(
                [e[:, :min_t_ecg] for e in all_ecg],
                axis=0,
            )

        if all_gsr:
            min_t_gsr = min(e.shape[1] for e in all_gsr)
            result["gsr"] = np.stack(
                [e[:, :min_t_gsr] for e in all_gsr],
                axis=0,
            )

        return result
