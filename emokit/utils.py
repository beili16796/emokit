# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Shared utilities for reproducibility, logging, and path resolution."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set global random seed for reproducibility across numpy, torch, and stdlib.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Global seed set to %d", seed)


def get_data_root() -> Path:
    """Return the root data directory, configurable via ``EMOKIT_DATA_ROOT``.

    Returns:
        Resolved ``pathlib.Path`` to the data directory.
    """
    return Path(os.environ.get("EMOKIT_DATA_ROOT", Path.home() / "emokit_data"))


class EmoKitError(Exception):
    """Base exception for all EmoKit errors."""


class EmoKitDataError(EmoKitError):
    """Raised when data loading or preprocessing fails."""


class EmoKitModelError(EmoKitError):
    """Raised when model construction, training, or inference fails."""


class EmoKitConfigError(EmoKitError):
    """Raised when configuration is invalid or missing required fields."""


class EmoKitFeatureError(EmoKitError):
    """Raised when feature extraction encounters an error."""
