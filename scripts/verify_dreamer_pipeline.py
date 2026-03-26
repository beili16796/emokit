# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Sanity check for the DREAMER dataset loader.

Run from repo root after ``pip install -e .``::

    python scripts/verify_dreamer_pipeline.py --root /data/ssd/xwt/DREAMER

If data are missing, the script exits with a clear message.
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
from emokit.utils import EmoKitDataError, set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Load DREAMER for a few subjects, check shapes, compute DE."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default="/data/ssd/xwt/DREAMER",
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
            "DREAMER data not available: %s\n"
            "Download DREAMER.mat from Zenodo and place under --root.",
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

        eeg = raw["eeg"]
        labels = raw["labels"]

        logger.info(
            "  EEG shape: %s  (n_windows=%d, channels=%d, samples=%d)",
            eeg.shape,
            eeg.shape[0],
            eeg.shape[1],
            eeg.shape[2],
        )

        assert eeg.shape[1] == 14, f"Expected 14 EEG channels, got {eeg.shape[1]}"
        assert (
            eeg.shape[0] >= 18
        ), f"Expected >=18 windows (18 videos), got {eeg.shape[0]}"

        label_counts = dict(zip(*np.unique(labels, return_counts=True)))
        logger.info("  Labels: %s", label_counts)

        # Compute DE features
        de = DEExtractor(fs=128)
        de_features = de.fit_transform(eeg)
        logger.info("  DE features shape: %s", de_features.shape)
        assert de_features.shape == (
            eeg.shape[0],
            14,
            5,
        ), f"DE shape mismatch: {de_features.shape}"

        ecg = raw.get("ecg")
        ecg_shape = tuple(ecg.shape) if ecg is not None else None

        summary[str(sid)] = {
            "eeg_shape": list(eeg.shape),
            "ecg_shape": list(ecg_shape) if ecg_shape else None,
            "de_shape": list(de_features.shape),
            "n_windows": int(eeg.shape[0]),
            "label_counts": {str(int(k)): int(v) for k, v in label_counts.items()},
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    logger.info("DREAMER pipeline verification complete.")


if __name__ == "__main__":
    main()
