# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Sanity check for the DREAMER dataset loader.

Run from repo root after ``pip install -e .``::

    python scripts/verify_dreamer_pipeline.py --root data/DREAMER

If data are missing, the script exits with a clear message instead of a traceback.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from emokit.datasets.dreamer import DREAMERDataset
from emokit.features.eeg import DEExtractor
from emokit.utils import EmoKitDataError, get_data_root, set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Load DREAMER for a few subjects, check shapes, and optionally compute DE."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default=str(get_data_root() / "DREAMER"),
        help="Path to directory containing DREAMER.mat.",
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--out",
        type=str,
        default="results/dreamer_sanity.json",
        help="Where to write JSON summary.",
    )
    args = parser.parse_args()

    set_seed(42)
    root = Path(args.root)

    try:
        ds = DREAMERDataset(
            root=str(root),
            subjects=args.subjects,
            label_axis="valence",
        )
    except (FileNotFoundError, EmoKitDataError, OSError) as exc:
        logger.error(
            "DREAMER data not available or path is wrong: %s\n"
            "Please download DREAMER.mat from Zenodo and place it under --root.",
            exc,
        )
        sys.exit(2)

    summary: dict[str, dict] = {}

    for sid in args.subjects:
        logger.info("--- Subject %d ---", sid)
        try:
            raw = ds.read_raw(sid)
        except EmoKitDataError as exc:
            logger.error("Failed to load subject %d: %s", sid, exc)
            continue

        eeg = raw.get("eeg")
        ecg = raw.get("ecg")
        labels = raw["labels"]

        eeg_shape = tuple(eeg.shape) if eeg is not None else None
        ecg_shape = tuple(ecg.shape) if ecg is not None else None
        unique_labels = sorted(set(labels.tolist()))

        logger.info(
            "Subject %d: EEG%s ECG%s Labels binary %s",
            sid,
            eeg_shape,
            ecg_shape,
            set(unique_labels),
        )

        summary[str(sid)] = {
            "eeg_shape": list(eeg_shape) if eeg_shape else None,
            "ecg_shape": list(ecg_shape) if ecg_shape else None,
            "n_trials": int(labels.shape[0]),
            "label_counts": {
                str(int(k)): int(v)
                for k, v in zip(*np.unique(labels, return_counts=True))
            },
        }

        if eeg is not None:
            assert eeg.shape[0] == 18, (
                f"Expected 18 trials, got {eeg.shape[0]}"
            )
            assert eeg.shape[1] == 14, (
                f"Expected 14 EEG channels, got {eeg.shape[1]}"
            )
            logger.info("  EEG assertions passed (18 trials, 14 channels)")

        if ecg is not None:
            assert ecg.shape[1] == 2, (
                f"Expected 2 ECG channels, got {ecg.shape[1]}"
            )
            logger.info("  ECG assertions passed (2 channels)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    logger.info("DREAMER pipeline verification complete.")


if __name__ == "__main__":
    main()
