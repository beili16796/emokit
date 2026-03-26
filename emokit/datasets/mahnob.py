# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""MAHNOB-HCI dataset loader (32-channel EEG + ECG + GSR).

Supports two on-disk layouts:

1. **CSV** (local): ``eeg/{subject_id}/{trial_id}_eeg.csv`` with matching
   ``ecg/`` and ``gsr/`` directories, plus ``valence_label.csv`` and
   ``arousal_label.csv`` at the root.
2. **BDF** (original): Per-session ``.bdf`` files in subject directories.
"""

from __future__ import annotations

import csv
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
_MISSING_SUBJECTS: frozenset[int] = frozenset({12, 15, 26})


@_REGISTRY.register("MAHNOB-HCI")
class MAHNOBHCIDataset(BaseDataset):
    """MAHNOB-HCI: Multimodal Affect-HCI dataset.

    Reference:
        Soleymani et al., *IEEE Trans. Affective Computing*, 2012.

    Args:
        root: Path to the MAHNOB-HCI data directory.
        subjects: Subject IDs to load (1-based).
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
        all_ids = set(range(1, 31)) - _MISSING_SUBJECTS
        eeg_dir = self.root / "eeg"
        if eeg_dir.is_dir():
            on_disk = {
                int(d.name)
                for d in eeg_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            }
            all_ids = all_ids & on_disk
        return sorted(all_ids)

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

    # ------------------------------------------------------------------
    # CSV layout detection and loading
    # ------------------------------------------------------------------

    def _is_csv_layout(self) -> bool:
        return (self.root / "eeg").is_dir()

    def _load_csv_labels(self) -> dict[int, np.ndarray]:
        """Load subject → trial labels from ``{label_axis}_label.csv``.

        CSV layout: rows = subjects (0-indexed), columns = trials (0-indexed),
        first row and first column are headers.

        Returns:
            Mapping from subject_id (int) to 1D binary label array.
        """
        csv_name = f"{self.label_axis}_label.csv"
        path = self.root / csv_name
        if not path.exists():
            raise EmoKitDataError(f"Label file not found: {path}")

        with open(path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header row
            labels_map: dict[int, np.ndarray] = {}
            for row in reader:
                sid = int(row[0])
                vals = []
                for v in row[1:]:
                    v = v.strip()
                    if v == "" or v == "nan":
                        vals.append(0)
                    else:
                        vals.append(1 if float(v) > self.label_threshold else 0)
                labels_map[sid] = np.array(vals, dtype=np.int64)
        return labels_map

    def _load_csv_signal(
        self,
        subject_id: int,
        modality: str,
    ) -> list[np.ndarray]:
        """Load all trial CSVs for one subject and modality.

        Returns:
            List of 1D or 2D arrays, one per trial (channels × samples).
        """
        mod_dir = self.root / modality / str(subject_id)
        if not mod_dir.is_dir():
            return []

        trial_files = sorted(
            mod_dir.glob(f"*_{modality}.csv"),
            key=lambda p: int(p.stem.split("_")[0]),
        )

        arrays: list[np.ndarray] = []
        for tf in trial_files:
            with open(tf) as f:
                reader = csv.reader(f)
                next(reader)  # skip header row
                data = []
                for row in reader:
                    data.append([float(v) for v in row[1:]])  # skip index col
            arr = np.array(data, dtype=np.float64).T  # (channels, samples)
            arrays.append(arr)
        return arrays

    def _read_csv_layout(self, subject_id: int) -> dict[str, np.ndarray]:
        """Load from CSV directory layout."""
        labels_map = self._load_csv_labels()
        subject_labels = labels_map.get(subject_id - 1)  # CSV uses 0-indexed rows
        if subject_labels is None:
            subject_labels = labels_map.get(subject_id, np.array([], dtype=np.int64))

        eeg_trials = self._load_csv_signal(subject_id, "eeg")
        if not eeg_trials:
            raise EmoKitDataError(
                f"No EEG CSV files found for HCI subject {subject_id}"
            )

        n_channels_eeg = 32
        processed_eeg: list[np.ndarray] = []
        for trial_arr in eeg_trials:
            n_ch = min(trial_arr.shape[0], n_channels_eeg)
            eeg = trial_arr[:n_ch, :]

            # Estimate fs from sample count (~11k samples for ~86s recording)
            n_samples = eeg.shape[1]
            estimated_fs = max(128.0, round(n_samples / 86.0))
            if estimated_fs > 200:
                estimated_fs = 256.0

            sos = butter(5, [1.0, 45.0], btype="band", fs=estimated_fs, output="sos")
            eeg = sosfiltfilt(sos, eeg, axis=-1)
            eeg -= eeg.mean(axis=0, keepdims=True)

            down = int(estimated_fs / _TARGET_FS) if estimated_fs > _TARGET_FS else 1
            if down > 1:
                eeg = resample_poly(eeg, up=1, down=down, axis=-1)

            processed_eeg.append(eeg)

        min_t = min(e.shape[1] for e in processed_eeg)
        eeg_arr = np.stack([e[:, :min_t] for e in processed_eeg], axis=0)

        n_trials = eeg_arr.shape[0]
        if len(subject_labels) >= n_trials:
            labels = subject_labels[:n_trials]
        else:
            labels = np.zeros(n_trials, dtype=np.int64)

        result: dict[str, np.ndarray] = {
            "eeg": eeg_arr,
            "labels": labels,
        }

        # ECG
        ecg_trials = self._load_csv_signal(subject_id, "ecg")
        if ecg_trials:
            ecg_processed: list[np.ndarray] = []
            for arr in ecg_trials:
                down = int(256 / _TARGET_FS) if 256 > _TARGET_FS else 1
                ecg = arr[:1, :] if arr.ndim == 2 else arr[np.newaxis, :]
                if down > 1:
                    ecg = resample_poly(ecg, up=1, down=down, axis=-1)
                ecg_processed.append(ecg)
            min_t_ecg = min(e.shape[1] for e in ecg_processed)
            result["ecg"] = np.stack(
                [e[:, :min_t_ecg] for e in ecg_processed[:n_trials]],
                axis=0,
            )

        # GSR
        gsr_trials = self._load_csv_signal(subject_id, "gsr")
        if gsr_trials:
            gsr_processed: list[np.ndarray] = []
            for arr in gsr_trials:
                down = int(256 / _TARGET_FS) if 256 > _TARGET_FS else 1
                gsr = arr[:1, :] if arr.ndim == 2 else arr[np.newaxis, :]
                if down > 1:
                    gsr = resample_poly(gsr, up=1, down=down, axis=-1)
                gsr_processed.append(gsr)
            min_t_gsr = min(e.shape[1] for e in gsr_processed)
            result["gsr"] = np.stack(
                [e[:, :min_t_gsr] for e in gsr_processed[:n_trials]],
                axis=0,
            )

        return result

    # ------------------------------------------------------------------
    # BDF layout (fallback)
    # ------------------------------------------------------------------

    def _find_session_dirs(self, subject_id: int) -> list[Path]:
        subject_dir = self.root / f"Subject{subject_id}"
        if not subject_dir.exists():
            subject_dir = self.root / f"s{subject_id:02d}"
        if not subject_dir.exists():
            raise EmoKitDataError(
                f"Subject directory not found for subject {subject_id} "
                f"under {self.root}"
            )
        return sorted(
            p for p in subject_dir.iterdir() if p.is_dir() or p.suffix == ".bdf"
        )

    def _load_bdf(self, bdf_path: Path) -> tuple[np.ndarray, float]:
        try:
            import mne
        except ImportError as exc:
            raise EmoKitDataError(
                "MNE is required for .bdf loading. Install with: pip install mne"
            ) from exc
        raw = mne.io.read_raw_bdf(str(bdf_path), preload=True, verbose=False)
        return raw.get_data(), raw.info["sfreq"]

    def _read_bdf_layout(self, subject_id: int) -> dict[str, np.ndarray]:
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
                    continue
                bdf_file = bdf_candidates[0]

            data, sfreq = self._load_bdf(bdf_file)
            sos = butter(5, [1.0, 45.0], btype="band", fs=sfreq, output="sos")
            eeg = sosfiltfilt(sos, data[:32, :], axis=-1)
            eeg -= eeg.mean(axis=0, keepdims=True)

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
        result: dict[str, np.ndarray] = {
            "eeg": np.stack([e[:, :min_t] for e in all_eeg], axis=0),
            "labels": np.asarray(all_labels, dtype=np.int64),
        }
        if all_ecg:
            min_t_ecg = min(e.shape[1] for e in all_ecg)
            result["ecg"] = np.stack([e[:, :min_t_ecg] for e in all_ecg], axis=0)
        if all_gsr:
            min_t_gsr = min(e.shape[1] for e in all_gsr)
            result["gsr"] = np.stack([e[:, :min_t_gsr] for e in all_gsr], axis=0)
        return result

    # ------------------------------------------------------------------
    # BaseDataset interface
    # ------------------------------------------------------------------

    def read_raw(self, subject_id: int) -> dict[str, np.ndarray]:
        """Load MAHNOB-HCI data for one subject.

        Auto-detects CSV layout (``eeg/`` directory present) vs BDF layout.
        """
        if self._is_csv_layout():
            return self._read_csv_layout(subject_id)
        return self._read_bdf_layout(subject_id)
