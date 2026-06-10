#!/usr/bin/env python3
"""Verify DREAMER data loading pipeline with the real DREAMER.mat file.

Usage::

    python -m emokit.scripts.verify_dreamer_pipeline --root /data/ssd/xwt/DREAMER
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from emokit.datasets.dreamer import DREAMERDataset

logger = logging.getLogger(__name__)

_N_EEG_CHANNELS = 14
_N_ECG_CHANNELS = 2
_N_VIDEOS = 18
_FS = 128
_WINDOW_SEC = 4.0


def verify(root: str, subject_id: int) -> None:
    root_path = Path(root)
    mat_path = root_path / "DREAMER.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"DREAMER.mat not found at {mat_path}")
    print(f"Found: {mat_path} ({mat_path.stat().st_size / 1024 / 1024:.1f} MB)")

    t0 = time.perf_counter()
    ds = DREAMERDataset(root=root, subjects=[subject_id], label_axis="valence")
    data = ds.read_raw(subject_id)
    elapsed = time.perf_counter() - t0
    print(f"Loaded subject {subject_id} in {elapsed:.2f}s")

    eeg = data.get("eeg")
    ecg = data.get("ecg")
    labels = data["labels"]

    print(f"\nEEG shape:    {eeg.shape if eeg is not None else 'MISSING'}")
    print(f"ECG shape:    {ecg.shape if ecg is not None else 'MISSING'}")
    print(f"Labels shape: {labels.shape}")

    errors: list[str] = []

    # --- EEG checks ---
    if eeg is None:
        errors.append("EEG data is missing")
    else:
        if eeg.shape[0] != _N_VIDEOS:
            errors.append(f"Expected {_N_VIDEOS} trials, got {eeg.shape[0]}")
        if eeg.shape[1] != _N_EEG_CHANNELS:
            errors.append(
                f"Expected {_N_EEG_CHANNELS} EEG channels, got {eeg.shape[1]}"
            )
        if np.any(np.isnan(eeg)):
            errors.append("EEG contains NaN values")
        if np.any(np.isinf(eeg)):
            errors.append("EEG contains Inf values")
        print(f"EEG range:    [{float(eeg.min()):.4f}, {float(eeg.max()):.4f}]")

    # --- ECG checks ---
    if ecg is None:
        errors.append("ECG data is missing")
    else:
        if ecg.shape[0] != _N_VIDEOS:
            errors.append(f"Expected {_N_VIDEOS} ECG trials, got {ecg.shape[0]}")
        if ecg.shape[1] != _N_ECG_CHANNELS:
            errors.append(
                f"Expected {_N_ECG_CHANNELS} ECG channels, got {ecg.shape[1]}"
            )
        if np.any(np.isnan(ecg)):
            errors.append("ECG contains NaN values")

        # ECG and EEG should have comparable time dimensions after downsampling
        if eeg is not None and abs(ecg.shape[2] - eeg.shape[2]) > 5:
            errors.append(
                "EEG/ECG time mismatch after downsample: "
                f"EEG={eeg.shape[2]}, ECG={ecg.shape[2]}"
            )

    # --- Label checks ---
    if labels.ndim != 1:
        errors.append(f"Labels should be 1D, got {labels.ndim}D")
    if len(labels) != _N_VIDEOS:
        errors.append(f"Expected {_N_VIDEOS} labels, got {len(labels)}")
    unique_labels = set(labels.tolist())
    if not unique_labels.issubset({0, 1}):
        errors.append(f"Labels should be binary {{0,1}}, got {unique_labels}")

    print(
        "Label distribution: "
        f"0={int((labels == 0).sum())}, 1={int((labels == 1).sum())}"
    )

    # --- Windowing check ---
    print(f"\nWindowing test ({_WINDOW_SEC}s window, 50% overlap):")
    X, y = ds.load()
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    win_samples = int(_WINDOW_SEC * _FS)
    # load() concatenates all modalities: EEG(14) + ECG(2) = 16
    expected_channels = _N_EEG_CHANNELS + _N_ECG_CHANNELS
    if X.shape[1] != expected_channels:
        errors.append(
            f"Windowed X has {X.shape[1]} channels, expected {expected_channels}"
        )
    if X.shape[2] != win_samples:
        errors.append(f"Windowed X has {X.shape[2]} samples, expected {win_samples}")
    if np.any(np.isnan(X)):
        errors.append("Windowed X contains NaN")

    print(
        f"  Subject {subject_id}: EEG({X.shape[0]}, {X.shape[1]}, {X.shape[2]}) "
        f"Labels{{0:{int((y == 0).sum())}, 1:{int((y == 1).sum())}}} "
        f"{'No NaN' if not np.any(np.isnan(X)) else 'HAS NaN!'}"
    )

    if errors:
        print(f"\n{'='*60}")
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        raise RuntimeError(
            f"DREAMER pipeline verification failed with {len(errors)} errors"
        )

    print(f"\n{'='*60}")
    print("ALL CHECKS PASSED")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Path to DREAMER data directory")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1-23)")
    args = parser.parse_args()
    verify(args.root, args.subject)


if __name__ == "__main__":
    main()
