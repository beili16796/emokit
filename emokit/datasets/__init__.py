# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""EmoKit datasets — unified loaders for public affective computing corpora."""

from __future__ import annotations

from emokit.datasets.base import (
    BaseDataset,
    DatasetRegistry,
    _REGISTRY,
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
