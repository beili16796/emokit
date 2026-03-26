# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Feature extraction transforms and pipeline utilities."""

from __future__ import annotations

from emokit.features.augmentation import (
    FeatureMixup,
    TemporalSegmentPermutation,
)
from emokit.features.base import (
    GLOBAL_REGISTRY,
    BaseTransform,
    FeaturePipeline,
    TransformRegistry,
)
from emokit.features.channel_align import (
    align_channels,
    subset_features,
)
from emokit.features.eeg import (
    BandpowerExtractor,
    DEExtractor,
    EEGNormalizer,
)
from emokit.features.peripheral import (
    GSRExtractor,
    HRVExtractor,
    ModalityFusionTransform,
)

__all__ = [
    "BaseTransform",
    "BandpowerExtractor",
    "DEExtractor",
    "EEGNormalizer",
    "FeatureMixup",
    "FeaturePipeline",
    "GLOBAL_REGISTRY",
    "GSRExtractor",
    "HRVExtractor",
    "ModalityFusionTransform",
    "TemporalSegmentPermutation",
    "TransformRegistry",
    "align_channels",
    "subset_features",
]
