# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""10-20 system channel alignment for cross-corpus evaluation.

Provides a canonical mapping from dataset-specific channel lists to the
International 10-20 electrode naming convention, enabling automatic
channel subsetting when source and target corpora have different montages.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

CANONICAL_10_20: dict[str, tuple[float, float]] = {
    "Fp1": (-0.31, 0.95),
    "Fp2": (0.31, 0.95),
    "AF3": (-0.41, 0.82),
    "AF4": (0.41, 0.82),
    "F7": (-0.81, 0.59),
    "F3": (-0.55, 0.59),
    "Fz": (0.00, 0.59),
    "F4": (0.55, 0.59),
    "F8": (0.81, 0.59),
    "FC5": (-0.69, 0.40),
    "FC1": (-0.22, 0.40),
    "FC2": (0.22, 0.40),
    "FC6": (0.69, 0.40),
    "T7": (-1.00, 0.00),
    "C3": (-0.55, 0.00),
    "Cz": (0.00, 0.00),
    "C4": (0.55, 0.00),
    "T8": (1.00, 0.00),
    "CP5": (-0.69, -0.40),
    "CP1": (-0.22, -0.40),
    "CP2": (0.22, -0.40),
    "CP6": (0.69, -0.40),
    "P7": (-0.81, -0.59),
    "P3": (-0.55, -0.59),
    "Pz": (0.00, -0.59),
    "P4": (0.55, -0.59),
    "P8": (0.81, -0.59),
    "PO3": (-0.41, -0.82),
    "PO4": (0.41, -0.82),
    "O1": (-0.31, -0.95),
    "Oz": (0.00, -0.95),
    "O2": (0.31, -0.95),
}

_ALIASES: dict[str, str] = {
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",
}


def _normalise(name: str) -> str:
    """Normalise a channel name to its canonical 10-20 form."""
    s = name.strip().upper()
    return _ALIASES.get(s, s).capitalize() if s else name


def align_channels(
    source_names: list[str],
    target_names: list[str],
) -> list[int]:
    """Find source channel indices that match the target montage.

    Uses the 10-20 naming convention for matching.  Returns the
    integer indices into *source_names* whose normalised names appear
    in *target_names*.

    Args:
        source_names: Channel names of the source corpus.
        target_names: Channel names of the target corpus.

    Returns:
        List of indices into *source_names*.  The ordering matches the
        target list (first index corresponds to target_names[0], etc.).

    Raises:
        ValueError: If any target channel cannot be found in source.
    """
    src_map = {_normalise(n): i for i, n in enumerate(source_names)}
    indices: list[int] = []
    for tgt in target_names:
        key = _normalise(tgt)
        if key not in src_map:
            raise ValueError(
                f"Target channel '{tgt}' ({key}) not found in "
                f"source montage ({list(src_map.keys())[:10]}...)"
            )
        indices.append(src_map[key])
    return indices


def subset_features(
    X: np.ndarray,
    source_names: list[str],
    target_names: list[str],
) -> np.ndarray:
    """Select channel subset from a feature array.

    Works with both raw ``(N, C, T)`` and DE ``(N, C, bands)`` arrays
    where the second axis is the channel axis.

    Args:
        X: Feature array with channels along axis 1.
        source_names: Source channel names (length == X.shape[1]).
        target_names: Desired channel subset.

    Returns:
        Array with only the matched channels.
    """
    idx = align_channels(source_names, target_names)
    return X[:, idx]
