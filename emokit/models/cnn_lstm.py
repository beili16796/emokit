# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""CNN-LSTM hybrid model for raw EEG or pre-extracted DE features."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from emokit.models.base import BaseModel, StandardTrainer, registry

logger = logging.getLogger(__name__)


class _CNNLSTM(nn.Module):
    """Internal CNN-LSTM network.

    Args:
        n_classes: Number of output classes.
        input_type: ``'raw'`` for (batch, C, T) or ``'de'`` for (batch, C*5).
        n_channels: Number of EEG channels (used for raw input).
        hidden_size: LSTM hidden dimension.
        n_layers: Number of BiLSTM layers.
        dropout: Dropout probability for LSTM.
    """

    def __init__(
        self,
        n_classes: int,
        input_type: str = "raw",
        n_channels: int = 62,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_type = input_type

        if input_type == "raw":
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=(1, 5), padding=(0, 2)),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, 4)),
                nn.Conv2d(64, 128, kernel_size=(1, 3), padding=(0, 1)),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
            self._lstm_input_dim = 128
        else:
            self.cnn = None
            self._lstm_input_dim = n_channels * 5

        self.lstm = nn.LSTM(
            input_size=self._lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor. Shape depends on ``input_type``.

        Returns:
            Logits of shape ``(batch, n_classes)``.
        """
        if self.input_type == "raw":
            assert x.ndim == 3, f"Expected 3-D input (B, C, T) for raw, got {x.shape}"
            x = x.unsqueeze(1)  # (B, 1, C, T)
            x = self.cnn(x)  # (B, 128, C, T')
            b, c, ch, t = x.shape
            x = x.permute(0, 3, 2, 1).reshape(b, t, ch * c)  # (B, T', C*128)
            # Take mean over channel dim to get (B, T', 128)
            x = x.reshape(b, t, ch, c).mean(dim=2)  # (B, T', 128)
        else:
            assert x.ndim == 2, f"Expected 2-D input (B, C*5) for de, got {x.shape}"
            x = x.unsqueeze(1)  # (B, 1, C*5)

        out, _ = self.lstm(x)  # (B, seq, 2*hidden)
        out = out[:, -1, :]  # last time step
        return self.fc(out)


@registry.register("CNN-LSTM")
class CNNLSTMModel(BaseModel):
    """CNN-LSTM model for EEG-based emotion recognition.

    Supports raw EEG waveforms or pre-extracted differential entropy (DE)
    features.  Controlled by the ``input_type`` config key.

    Config keys:
        n_classes, lr, batch_size, n_epochs, input_type, hidden_size,
        n_layers, dropout, device.
    """

    def __init__(
        self,
        n_classes: int = 3,
        lr: float = 1e-3,
        batch_size: int = 64,
        n_epochs: int = 100,
        input_type: str = "raw",
        n_channels: int = 62,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(n_classes=n_classes, device=device)
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.input_type = input_type
        self.n_channels = n_channels
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.seed = seed
        self._build()

    def _build(self) -> None:
        self.network = _CNNLSTM(
            n_classes=self.n_classes,
            input_type=self.input_type,
            n_channels=self.n_channels,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Train the CNN-LSTM model.

        Args:
            X_train: Training data, shape ``(N, C, T)`` for raw or ``(N, C*5)`` for DE.
            y_train: Training labels.
            X_val: Optional validation data.
            y_val: Optional validation labels.

        Returns:
            Training history dict.
        """
        trainer = StandardTrainer(
            n_epochs=self.n_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            patience=10,
            device=str(self.device),
            seed=self.seed,
        )
        return trainer.train(
            model=self.network,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(X, dtype=torch.float32).to(self.device)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions.

        Args:
            X: Input features.

        Returns:
            Predicted class indices.
        """
        self.network.eval()
        logits = self.network(self._to_tensor(X))
        return logits.argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities.

        Args:
            X: Input features.

        Returns:
            Probability array of shape ``(N, n_classes)``.
        """
        self.network.eval()
        logits = self.network(self._to_tensor(X))
        return torch.softmax(logits, dim=-1).cpu().numpy()
