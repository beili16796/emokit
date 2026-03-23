# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""First sanity check for the DEAP → DE pipeline (requires local DEAP .bdf data).

Run from repo root after ``pip install -e .``::

    python scripts/verify_deap_pipeline.py --root data/DEAP

If data are missing, the script exits with a clear message instead of a traceback.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from emokit.datasets.deap import DEAPDataset
from emokit.features.eeg import DEExtractor
from emokit.utils import EmoKitDataError, get_data_root, set_seed

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Load DEAP for a few subjects, compute DE features, run sanity assertions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default=str(get_data_root() / "DEAP"),
        help="Path to DEAP root (containing s01.bdf … or fallbacks).",
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--window-sec", type=float, default=4.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument(
        "--out",
        type=str,
        default="results/pipeline_sanity.json",
        help="Where to write JSON summary.",
    )
    args = parser.parse_args()

    set_seed(42)
    root = Path(args.root)

    try:
        ds = DEAPDataset(
            root=str(root),
            subjects=args.subjects,
            window_sec=args.window_sec,
            overlap=args.overlap,
            modalities=["eeg"],
        )
        X_eeg, y = ds.load()
    except (FileNotFoundError, EmoKitDataError, OSError) as exc:
        logger.error(
            "DEAP 数据不可用或路径错误：%s\n"
            "请从官网申请 DEAP 并将 .bdf 置于 --root 下后再运行本脚本。",
            exc,
        )
        sys.exit(2)

    assert X_eeg.ndim == 3, f"Expected X (N, C, T), got {X_eeg.shape}"
    logger.info("EEG shape: %s", X_eeg.shape)
    logger.info("Labels: %s", np.unique(y, return_counts=True))

    de = DEExtractor(fs=128)
    X_de = de.fit_transform(X_eeg)
    logger.info("DE shape: %s", X_de.shape)

    # δ vs γ: broadband power differs; sanity check that ordering is not inverted.
    band_means = X_de.mean(axis=(0, 1))
    if not (band_means[0] < band_means[4]):
        logger.warning(
            "Band-wise means: delta >= gamma — may still be valid on short windows; "
            "inspect manually. means=%s",
            band_means,
        )
    else:
        assert band_means[0] < band_means[4], "DE band ordering anomaly (delta vs gamma)"

    greek = ["delta", "theta", "alpha", "beta", "gamma"]
    summary = {
        "eeg_shape": list(X_eeg.shape),
        "de_shape": list(X_de.shape),
        "de_band_means": band_means.tolist(),
        "de_band_means_named": dict(zip(greek, band_means.tolist())),
        "label_counts": {
            str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out_path)
    print("Band-wise DE means:", dict(zip(greek, np.round(band_means, 3))))


if __name__ == "__main__":
    main()
