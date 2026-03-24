# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Sanity check for the MAHNOB-HCI dataset loader.

Run from repo root after ``pip install -e .``::

    python scripts/verify_hci_pipeline.py --root data/MAHNOB-HCI

If data are missing, the script exits with a clear message instead of a traceback.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from emokit.datasets.mahnob import MAHNOBHCIDataset
from emokit.utils import EmoKitDataError, get_data_root, set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Load MAHNOB-HCI for a few subjects and verify shapes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default=str(get_data_root() / "MAHNOB-HCI"),
        help="Path to MAHNOB-HCI data directory.",
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--out",
        type=str,
        default="results/hci_sanity.json",
        help="Where to write JSON summary.",
    )
    args = parser.parse_args()

    set_seed(42)
    root = Path(args.root)

    try:
        ds = MAHNOBHCIDataset(
            root=str(root),
            subjects=args.subjects,
            modalities=["eeg"],
            label_axis="valence",
        )
    except (FileNotFoundError, EmoKitDataError, OSError) as exc:
        logger.error(
            "MAHNOB-HCI data not available or path is wrong: %s\n"
            "Please download the MAHNOB-HCI dataset and place it under --root.",
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
        labels = raw["labels"]

        if eeg is not None:
            eeg_shape = tuple(eeg.shape)
            assert eeg.shape[1] == 32, (
                f"Expected 32 EEG channels, got {eeg.shape[1]}"
            )
            assert not np.isnan(eeg[0]).any(), (
                f"NaN found in subject {sid}, trial 0 EEG"
            )

            label_val = int(labels[0])
            logger.info(
                "Subject %d, Trial 1: EEG%s Label valence=%d",
                sid,
                eeg_shape,
                label_val,
            )
        else:
            eeg_shape = None
            logger.warning("Subject %d: no EEG data found", sid)

        summary[str(sid)] = {
            "eeg_shape": list(eeg_shape) if eeg_shape else None,
            "n_trials": int(labels.shape[0]),
            "label_counts": {
                str(int(k)): int(v)
                for k, v in zip(*np.unique(labels, return_counts=True))
            },
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    logger.info("MAHNOB-HCI pipeline verification complete.")


if __name__ == "__main__":
    main()
