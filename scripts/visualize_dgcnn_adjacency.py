# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Visualise DGCNN learned adjacency (example: one electrode's row as topomap).

Requires a trained checkpoint from ``emokit``::

    python scripts/visualize_dgcnn_adjacency.py \\
        --checkpoint checkpoints/dgcnn_deap_valence.pt \\
        --channels 32 \\
        --out figures/dgcnn_adjacency_row.pdf

Channel names default to DEAP 32-channel 10-20 order used in the toolkit.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402

from emokit.models.dgcnn import DGCNNModel

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEAP_CHANNEL_NAMES: list[str] = [
    "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7",
    "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
    "Fp2", "AF4", "F4", "F8", "FC6", "FC2", "C4", "T8",
    "CP6", "CP2", "P4", "P8", "PO4", "O2", "Fz", "Cz",
]


def main() -> None:
    """Load DGCNN weights and plot one row of the adjacency matrix on the scalp."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--bands", type=int, default=5)
    parser.add_argument("--row-index", type=int, default=30, help="Electrode index (e.g. Fz≈30 in DEAP list).")
    parser.add_argument("--out", type=str, default="figures/dgcnn_adjacency_row.pdf")
    parser.add_argument("--montage", type=str, default="standard_1020")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        logger.error("Checkpoint not found: %s", ckpt)
        sys.exit(2)

    names = DEAP_CHANNEL_NAMES[: args.channels]
    if len(names) != args.channels:
        logger.error("Channel list length must match --channels.")
        sys.exit(2)

    model = DGCNNModel(
        n_classes=2,
        n_channels=args.channels,
        n_bands=args.bands,
        n_epochs=1,
    )
    model.load(str(ckpt))
    a = model.get_adjacency_matrix()
    row = a[args.row_index]

    info = mne.create_info(ch_names=names, sfreq=128.0, ch_types="eeg")
    montage = mne.channels.make_standard_montage(args.montage)
    info.set_montage(montage, on_missing="warn")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    mne.viz.plot_topomap(row, info, axes=ax, cmap="RdBu_r", show=False)
    ax.set_title(f"DGCNN adjacency row {args.row_index} ({names[args.row_index]})")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    logger.info("Saved %s", out)


if __name__ == "__main__":
    main()
