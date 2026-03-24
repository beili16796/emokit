#!/usr/bin/env python3
"""Verify SEED data loading pipeline with real .mat files.

Usage::

    python -m emokit.scripts.verify_seed_pipeline --root /path/to/SEED --subject 1
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from emokit.datasets.seed import SEEDDataset

logger = logging.getLogger(__name__)

_EXPECTED_N_EEG_CHANNELS = 62
_EXPECTED_N_CLASSES = 3
_LABEL_NAMES = ["negative", "neutral", "positive"]


def verify(root: str, subject_id: int, use_de: bool = True) -> None:
    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"SEED root not found: {root_path}")

    print(f"SEED root: {root_path}")
    print(f"Subject: {subject_id}, use_de_features={use_de}")

    t0 = time.perf_counter()
    ds = SEEDDataset(root=root, subjects=[subject_id], use_de_features=use_de)
    data = ds.read_raw(subject_id)
    elapsed = time.perf_counter() - t0
    print(f"Loaded in {elapsed:.2f}s")

    eeg = data["eeg"]
    labels = data["labels"]

    print(f"\nEEG shape:    {eeg.shape}")
    print(f"Labels shape: {labels.shape}")

    errors: list[str] = []

    if use_de:
        if eeg.ndim != 3:
            errors.append(f"DE features should be 3D (windows, channels, bands), got {eeg.ndim}D")
        elif eeg.shape[1] != _EXPECTED_N_EEG_CHANNELS:
            errors.append(f"Expected {_EXPECTED_N_EEG_CHANNELS} channels, got {eeg.shape[1]}")
        print(f"DE feature shape: (windows={eeg.shape[0]}, ch={eeg.shape[1]}, bands={eeg.shape[2]})")
        de_range = (float(eeg.min()), float(eeg.max()))
        print(f"DE range: [{de_range[0]:.4f}, {de_range[1]:.4f}]")
    else:
        if eeg.shape[1] != _EXPECTED_N_EEG_CHANNELS:
            errors.append(f"Expected {_EXPECTED_N_EEG_CHANNELS} EEG channels, got {eeg.shape[1]}")

    if np.any(np.isnan(eeg)):
        errors.append("EEG contains NaN values")
    if np.any(np.isinf(eeg)):
        errors.append("EEG contains Inf values")

    unique = sorted(set(labels.tolist()))
    if not all(0 <= v < _EXPECTED_N_CLASSES for v in unique):
        errors.append(f"Labels should be 0-{_EXPECTED_N_CLASSES - 1}, got unique={unique}")

    print(f"\nLabel distribution:")
    for i, name in enumerate(_LABEL_NAMES):
        count = int((labels == i).sum())
        print(f"  {i} ({name}): {count}")

    total = len(labels)
    print(f"\nTotal windows/trials: {total}")
    if total < 10:
        errors.append(f"Suspiciously few data points: {total}")

    if errors:
        print(f"\n{'='*60}")
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        raise RuntimeError(f"SEED pipeline verification failed with {len(errors)} errors")

    print(f"\n{'='*60}")
    print("ALL CHECKS PASSED")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Path to SEED data directory")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1-15)")
    parser.add_argument("--raw", action="store_true", help="Load raw EEG instead of DE features")
    args = parser.parse_args()
    verify(args.root, args.subject, use_de=not args.raw)


if __name__ == "__main__":
    main()
