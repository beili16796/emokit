#!/usr/bin/env python3
"""Verify DEAP data loading pipeline with real .dat files.

Usage::

    python -m emokit.scripts.verify_deap_pipeline --root /path/to/DEAP --subject 1
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from emokit.datasets.deap import DEAPDataset

logger = logging.getLogger(__name__)

_EXPECTED_N_TRIALS = 40
_EXPECTED_N_EEG_CHANNELS = 32
_EXPECTED_FS = 128
_BASELINE_SEC = 3.0
_TOTAL_RECORDING_SEC = 63.0  # 3s baseline + 60s trial
_EXPECTED_SAMPLES = int((_TOTAL_RECORDING_SEC - _BASELINE_SEC) * _EXPECTED_FS)  # 7680


def verify(root: str, subject_id: int) -> None:
    root_path = Path(root)
    mat_path = root_path / f"{subject_id}.mat"
    dat_path = root_path / f"s{subject_id:02d}.dat"
    bdf_path = root_path / f"s{subject_id:02d}.bdf"
    for p in (mat_path, dat_path, bdf_path):
        if p.exists():
            print(f"Found: {p} ({p.stat().st_size / 1024:.0f} KB)")
            break
    else:
        raise FileNotFoundError(
            f"No file found for subject {subject_id} in {root_path}. "
            f"Looked for: {mat_path}, {dat_path}, {bdf_path}"
        )

    t0 = time.perf_counter()
    ds = DEAPDataset(root=root, subjects=[subject_id], label_axis="valence")
    data = ds.read_raw(subject_id)
    elapsed = time.perf_counter() - t0
    print(f"Loaded in {elapsed:.2f}s")

    eeg = data.get("eeg")
    gsr = data.get("gsr")
    ecg = data.get("ecg")
    labels = data["labels"]

    print(f"\nEEG shape:    {eeg.shape if eeg is not None else 'MISSING'}")
    print(f"GSR shape:    {gsr.shape if gsr is not None else 'MISSING'}")
    print(f"ECG shape:    {ecg.shape if ecg is not None else 'MISSING'}")
    print(f"Labels shape: {labels.shape}")

    errors: list[str] = []

    if eeg is None:
        errors.append("EEG data is missing")
    else:
        if eeg.shape[0] != _EXPECTED_N_TRIALS:
            errors.append(f"Expected {_EXPECTED_N_TRIALS} trials, got {eeg.shape[0]}")
        if eeg.shape[1] != _EXPECTED_N_EEG_CHANNELS:
            errors.append(f"Expected {_EXPECTED_N_EEG_CHANNELS} EEG channels, got {eeg.shape[1]}")
        if eeg.shape[2] != _EXPECTED_SAMPLES:
            # Allow some tolerance for resampled data
            if abs(eeg.shape[2] - _EXPECTED_SAMPLES) > 100:
                errors.append(
                    f"Expected ~{_EXPECTED_SAMPLES} samples per trial, got {eeg.shape[2]}"
                )
            else:
                print(f"  Note: sample count {eeg.shape[2]} vs expected {_EXPECTED_SAMPLES}")

        if np.any(np.isnan(eeg)):
            errors.append("EEG contains NaN values")
        if np.any(np.isinf(eeg)):
            errors.append("EEG contains Inf values")

        de_range = (float(eeg.min()), float(eeg.max()))
        print(f"EEG range:    [{de_range[0]:.4f}, {de_range[1]:.4f}]")

    if labels.ndim != 1:
        errors.append(f"Labels should be 1D, got {labels.ndim}D")
    if len(labels) != _EXPECTED_N_TRIALS:
        errors.append(f"Expected {_EXPECTED_N_TRIALS} labels, got {len(labels)}")
    unique_labels = set(labels.tolist())
    if not unique_labels.issubset({0, 1}):
        errors.append(f"Labels should be binary {{0,1}}, got {unique_labels}")

    print(f"\nLabel distribution: 0={int((labels == 0).sum())}, 1={int((labels == 1).sum())}")

    if gsr is not None:
        if gsr.shape[0] != _EXPECTED_N_TRIALS or gsr.shape[1] != 1:
            errors.append(f"GSR shape unexpected: {gsr.shape}")
        if np.any(np.isnan(gsr)):
            errors.append("GSR contains NaN")
    else:
        print("  Warning: GSR not extracted (may be fine for EEG-only experiments)")

    if errors:
        print(f"\n{'='*60}")
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        raise RuntimeError(f"DEAP pipeline verification failed with {len(errors)} errors")

    print(f"\n{'='*60}")
    print("ALL CHECKS PASSED")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Path to DEAP data directory")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1-32)")
    args = parser.parse_args()
    verify(args.root, args.subject)


if __name__ == "__main__":
    main()
