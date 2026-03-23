# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""EmoKit model library — registry, base classes, and concrete model implementations."""

from __future__ import annotations

from emokit.models.base import (
    BaseModel,
    EarlyStopping,
    ModelRegistry,
    StandardTrainer,
    build_model,
    registry,
)
from emokit.models.bidae import BiDAEModel
from emokit.models.cnn_lstm import CNNLSTMModel
from emokit.models.dgcca_am import DGCCAAMModel
from emokit.models.dgcnn import DGCNNModel
from emokit.models.prpl import PRPLModel
from emokit.models.transformer_mm import TransformerMMModel

__all__ = [
    "BaseModel",
    "BiDAEModel",
    "CNNLSTMModel",
    "DGCCAAMModel",
    "DGCNNModel",
    "EarlyStopping",
    "ModelRegistry",
    "PRPLModel",
    "StandardTrainer",
    "TransformerMMModel",
    "build_model",
    "registry",
]
