#!/usr/bin/env python3
"""Generate high-fidelity mock data for all 5 EmoKit datasets.

Creates physically-realistic random data whose dimensions, sampling
rates, and file layouts match the real datasets documented in
``docs/dataset_setup.md``.  Band-limited Gaussian noise is injected
with plausible EEG/ECG/GSR power-spectral profiles so that downstream
feature extraction (DE, bandpower) produces numerically stable values
(no NaN / Inf).

Usage::

    python scripts/generate_mock_data.py --out /tmp/emokit_mock
    # then pass --deap-root /tmp/emokit_mock/DEAP  etc.
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Shared helpers ───────────────────────────────────────────────────


def _bandlimited_noise(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    fs: float,
    low: float = 1.0,
    high: float = 45.0,
) -> np.ndarray:
    """White Gaussian noise filtered to *low*–*high* Hz."""
    raw = rng.standard_normal(shape)
    if shape[-1] < 20:
        return raw * 0.1
    sos = butter(4, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, raw, axis=-1).astype(np.float64)


def _add_alpha_peak(
    data: np.ndarray, fs: float, rng: np.random.Generator
) -> np.ndarray:
    """Inject a 10 Hz alpha-band peak to mimic resting-state EEG."""
    t = np.arange(data.shape[-1]) / fs
    for ch in range(data.shape[-2] if data.ndim >= 2 else 1):
        amp = rng.uniform(0.3, 0.8)
        freq = rng.uniform(9.0, 11.0)
        if data.ndim == 3:
            data[:, ch, :] += amp * np.sin(2 * np.pi * freq * t)
        elif data.ndim == 2:
            data[ch, :] += amp * np.sin(2 * np.pi * freq * t)
    return data


# ── DEAP ─────────────────────────────────────────────────────────────


def generate_deap(out: Path, n_subjects: int = 3, seed: int = 0) -> None:
    """Generate mock DEAP .dat files (pickled)."""
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    fs = 128.0
    n_trials, n_channels, trial_samples = 40, 40, 8064

    for sid in range(1, n_subjects + 1):
        data = _bandlimited_noise(rng, (n_trials, n_channels, trial_samples), fs)
        data[:, :32, :] = _add_alpha_peak(data[:, :32, :], fs, rng)
        labels = np.column_stack(
            [
                rng.uniform(1, 9, n_trials),  # valence
                rng.uniform(1, 9, n_trials),  # arousal
                rng.uniform(1, 9, n_trials),  # dominance
                rng.uniform(1, 9, n_trials),  # liking
            ]
        )
        content = {"data": data, "labels": labels}
        path = out / f"s{sid:02d}.dat"
        with open(path, "wb") as f:
            pickle.dump(content, f, protocol=2)
        logger.info("DEAP: %s  data=%s", path.name, data.shape)


# ── SEED ─────────────────────────────────────────────────────────────


def generate_seed(out: Path, n_subjects: int = 3, seed: int = 100) -> None:
    """Generate mock SEED ExtractedFeatures .mat files."""
    from scipy.io import savemat

    ef_dir = out / "ExtractedFeatures"
    ef_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_sessions = 3
    n_trials_per_session = 15
    n_ch, n_bands = 62, 5

    label_path = ef_dir / "label.mat"
    session_labels = rng.integers(0, 3, size=n_trials_per_session)
    savemat(str(label_path), {"label": session_labels})

    names = [
        f"sub{sid}_{sess}"
        for sid in range(1, n_subjects + 1)
        for sess in range(1, n_sessions + 1)
    ]
    for name in names:
        mat_data = {}
        for t in range(1, n_trials_per_session + 1):
            n_win = rng.integers(10, 30)
            de = rng.standard_normal((n_ch, n_win, n_bands)) * 0.5
            mat_data[f"de_LDS{t}"] = de
        savemat(str(ef_dir / f"{name}.mat"), mat_data)
        logger.info("SEED: %s.mat (%d trials)", name, n_trials_per_session)


# ── SEED-V ───────────────────────────────────────────────────────────


def generate_seedv(out: Path, n_subjects: int = 3, seed: int = 200) -> None:
    """Generate mock SEED-V per-subject/session .mat files."""
    from scipy.io import savemat

    rng = np.random.default_rng(seed)
    n_sessions = 3
    n_trials = 15
    n_ch, n_bands = 62, 5

    for sid in range(1, n_subjects + 1):
        sdir = out / str(sid)
        sdir.mkdir(parents=True, exist_ok=True)
        for sess in range(1, n_sessions + 1):
            mat_data = {}
            for t in range(1, n_trials + 1):
                n_win = rng.integers(10, 30)
                mat_data[f"de_LDS{t}"] = (
                    rng.standard_normal((n_ch, n_win, n_bands)) * 0.5
                )
            mat_data["label"] = rng.integers(0, 5, size=n_trials)
            savemat(str(sdir / f"{sess}.mat"), mat_data)
        logger.info("SEED-V: subject %d (%d sessions)", sid, n_sessions)


# ── MAHNOB-HCI (CSV layout) ─────────────────────────────────────────


def generate_hci(out: Path, n_subjects: int = 3, seed: int = 300) -> None:
    """Generate mock MAHNOB-HCI CSV layout."""
    rng = np.random.default_rng(seed)
    n_trials = 20
    fs_eeg = 128
    trial_sec = 60

    # Label CSVs
    for axis in ("valence", "arousal"):
        rows = ["subject_id," + ",".join(str(t) for t in range(n_trials))]
        for sid in range(n_subjects):
            vals = rng.uniform(1, 9, n_trials)
            rows.append(f"{sid}," + ",".join(f"{v:.1f}" for v in vals))
        (out / f"{axis}_label.csv").write_text("\n".join(rows))

    for sid in range(1, n_subjects + 1):
        for mod, n_ch, fs in [("eeg", 32, fs_eeg), ("ecg", 1, 256), ("gsr", 1, 256)]:
            mod_dir = out / mod / str(sid)
            mod_dir.mkdir(parents=True, exist_ok=True)
            for t in range(1, n_trials + 1):
                n_samples = int(trial_sec * fs)
                header = "idx," + ",".join(f"ch{c}" for c in range(n_ch))
                data = _bandlimited_noise(rng, (n_ch, n_samples), float(fs))
                if mod == "eeg":
                    data = _add_alpha_peak(data, float(fs), rng)
                lines = [header]
                for s in range(n_samples):
                    row = f"{s}," + ",".join(f"{data[c, s]:.6f}" for c in range(n_ch))
                    lines.append(row)
                (mod_dir / f"{t}_{mod}.csv").write_text("\n".join(lines))
        logger.info("HCI: subject %d (%d trials)", sid, n_trials)


# ── DREAMER ──────────────────────────────────────────────────────────


def generate_dreamer(out: Path, n_subjects: int = 3, seed: int = 400) -> None:
    """Generate mock DREAMER.mat."""
    from scipy.io import savemat

    rng = np.random.default_rng(seed)
    n_videos = 18
    eeg_fs, ecg_fs = 128.0, 256.0

    class _Obj:
        pass

    subjects = []
    for _sid in range(n_subjects):
        subj = _Obj()
        subj.EEG = _Obj()
        subj.ECG = _Obj()

        eeg_stim, eeg_base = [], []
        ecg_stim, ecg_base = [], []
        val_scores, aro_scores = [], []

        for _v in range(n_videos):
            dur = rng.uniform(60, 200)
            m_eeg = int(dur * eeg_fs)
            m_ecg = int(dur * ecg_fs)
            m_bl = int(5 * eeg_fs)
            m_bl_ecg = int(5 * ecg_fs)

            eeg_s = _bandlimited_noise(rng, (m_eeg, 14), eeg_fs)
            eeg_s = _add_alpha_peak(eeg_s.T, eeg_fs, rng).T
            eeg_stim.append(eeg_s)
            eeg_base.append(_bandlimited_noise(rng, (m_bl, 14), eeg_fs))

            ecg_stim.append(rng.standard_normal((m_ecg, 2)) * 0.5)
            ecg_base.append(rng.standard_normal((m_bl_ecg, 2)) * 0.5)

            val_scores.append(rng.uniform(1, 5))
            aro_scores.append(rng.uniform(1, 5))

        subj.EEG.stimuli = np.array(eeg_stim, dtype=object)
        subj.EEG.baseline = np.array(eeg_base, dtype=object)
        subj.ECG.stimuli = np.array(ecg_stim, dtype=object)
        subj.ECG.baseline = np.array(ecg_base, dtype=object)
        subj.ScoreValence = np.array(val_scores)
        subj.ScoreArousal = np.array(aro_scores)
        subjects.append(subj)

    dreamer = _Obj()
    dreamer.Data = np.array(subjects, dtype=object)

    out.mkdir(parents=True, exist_ok=True)
    savemat(
        str(out / "DREAMER.mat"),
        {"DREAMER": dreamer},
        do_compression=True,
    )
    logger.info("DREAMER: %d subjects, %d videos each", n_subjects, n_videos)


# ── Main ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=str,
        default="/tmp/emokit_mock",
        help="Root output directory.",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=3,
        help="Subjects per dataset (default 3 for speed).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["deap", "seed", "seedv", "hci", "dreamer"],
        help="Which datasets to generate.",
    )
    args = parser.parse_args()
    root = Path(args.out)
    n = args.n_subjects

    generators = {
        "deap": lambda: generate_deap(root / "DEAP", n),
        "seed": lambda: generate_seed(root / "SEED", n),
        "seedv": lambda: generate_seedv(root / "SEED-V", n),
        "hci": lambda: generate_hci(root / "MAHNOB-HCI", n),
        "dreamer": lambda: generate_dreamer(root / "DREAMER", n),
    }

    for ds in args.datasets:
        if ds in generators:
            logger.info("=== Generating %s ===", ds.upper())
            generators[ds]()
        else:
            logger.warning("Unknown dataset: %s", ds)

    logger.info("All mock data written to %s", root)


if __name__ == "__main__":
    main()
