# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Visualise DGCNN learned adjacency as EEG topographic maps (paper Figure 2).

Usage::

    python -m emokit.scripts.visualize_dgcnn_adjacency \\
        --checkpoint checkpoints/dgcnn_deap_valence.pt \\
        --output figures/dgcnn_adjacency.pdf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from emokit.models.dgcnn import DGCNNModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEAP_CHANNEL_NAMES: list[str] = [
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


def plot_adjacency_topomap(
    checkpoint_path: Path,
    channel_names: list[str],
    output_path: Path,
    fs: int = 128,
) -> None:
    """Visualize DGCNN learned adjacency as EEG topographic map.

    For each reference electrode, plot its row in A as a scalp heatmap.
    Focus electrodes: F3, F4, Fz, Cz (frontal asymmetry relevant).
    """
    import mne

    model = DGCNNModel(n_classes=2, n_channels=len(channel_names), n_bands=5)
    model.load(str(checkpoint_path))
    A = model.get_adjacency_matrix()

    info = mne.create_info(ch_names=channel_names, sfreq=fs, ch_types="eeg")
    info.set_montage("standard_1020", match_case=False, on_missing="ignore")

    focus_channels = ["F3", "F4", "Fz", "Cz"]
    available = [ch for ch in focus_channels if ch in channel_names]
    n_focus = len(available)

    if n_focus == 0:
        logger.error("None of %s found in channel list.", focus_channels)
        return

    fig, axes = plt.subplots(1, n_focus, figsize=(3 * n_focus, 3))
    if n_focus == 1:
        axes = [axes]

    for ax, ch_name in zip(axes, available):
        ch_idx = channel_names.index(ch_name)
        values = A[ch_idx]
        mne.viz.plot_topomap(
            values,
            info,
            axes=ax,
            cmap="RdBu_r",
            show=False,
            sensors=True,
            contours=4,
        )
        ax.set_title(f"From {ch_name}", fontsize=10)

    fig.suptitle("DGCNN Learned Electrode Connectivity (A matrix rows)", fontsize=12)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=200, bbox_inches="tight")
    logger.info("Saved: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="figures/dgcnn_adjacency.pdf")
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--bands", type=int, default=5)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        logger.error(
            "No checkpoint found. Train DGCNN first with: "
            "python -m emokit.run configs/deap_loso_dgcnn.yaml"
        )
        sys.exit(2)

    names = DEAP_CHANNEL_NAMES[: args.channels]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    plot_adjacency_topomap(ckpt, names, out)


if __name__ == "__main__":
    main()
