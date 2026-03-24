# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""EmoKit datasets — unified loaders for public affective computing corpora."""

from __future__ import annotations

from emokit.datasets.base import (
    _REGISTRY,  # noqa: F401
    BaseDataset,
    DatasetRegistry,
    load_dataset,
    segment_trials,
)
from emokit.datasets.deap import DEAPDataset
from emokit.datasets.dreamer import DREAMERDataset
from emokit.datasets.mahnob import MAHNOBHCIDataset
from emokit.datasets.seed import SEEDDataset
from emokit.datasets.seedv import SEEDVDataset
from emokit.datasets.synthetic import SyntheticDataset

__all__ = [
    "_REGISTRY",
    "BaseDataset",
    "DatasetRegistry",
    "DEAPDataset",
    "DREAMERDataset",
    "MAHNOBHCIDataset",
    "SEEDDataset",
    "SEEDVDataset",
    "SyntheticDataset",
    "load_dataset",
    "segment_trials",
]
