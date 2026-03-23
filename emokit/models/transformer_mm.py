# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Multimodal Transformer model fusing EEG and peripheral physiological signals."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from emokit.models.base import BaseModel, EarlyStopping, registry

logger = logging.getLogger(__name__)


class _TransformerMM(nn.Module):
    """Internal multimodal Transformer network.

    Args:
        n_classes: Number of output classes.
        n_channels: Number of EEG channels.
        n_bands: Number of DE frequency bands.
        n_peripheral_feat: Dimension of peripheral feature vector.
        d_model: Transformer embedding dimension.
        nhead: Number of attention heads.
        n_layers: Number of Transformer encoder layers.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        n_classes: int,
        n_channels: int = 62,
        n_bands: int = 5,
        n_peripheral_feat: int = 7,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model

        self.eeg_proj = nn.Linear(n_bands, d_model)
        self.periph_proj = nn.Sequential(
            nn.Linear(n_peripheral_feat, d_model),
            nn.ReLU(),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(
        self, x_eeg: torch.Tensor, x_periph: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x_eeg: EEG DE features ``(batch, C, n_bands)``.
            x_periph: Peripheral features ``(batch, n_peripheral_feat)`` or None.

        Returns:
            Logits ``(batch, n_classes)``.
        """
        cls_out = self._encode(x_eeg, x_periph)
        return self.head(cls_out)

    def forward_features(
        self, x_eeg: torch.Tensor, x_periph: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return CLS embedding before the classification head.

        Args:
            x_eeg: EEG DE features ``(batch, C, n_bands)``.
            x_periph: Peripheral features or None.

        Returns:
            CLS embedding ``(batch, d_model)``.
        """
        return self._encode(x_eeg, x_periph)

    def _encode(
        self, x_eeg: torch.Tensor, x_periph: torch.Tensor | None = None
    ) -> torch.Tensor:
        B = x_eeg.size(0)
        eeg_tokens = self.eeg_proj(x_eeg)  # (B, C, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)

        if x_periph is not None:
            periph_token = self.periph_proj(x_periph).unsqueeze(1)  # (B, 1, d_model)
            tokens = torch.cat([cls_tokens, eeg_tokens, periph_token], dim=1)
        else:
            tokens = torch.cat([cls_tokens, eeg_tokens], dim=1)

        # NO positional encoding — Wu et al. 2024 Table VII ablation
        encoded = self.encoder(tokens)
        return encoded[:, 0, :]  # CLS readout


@registry.register("Transformer-MM")
class TransformerMMModel(BaseModel):
    """Multimodal Transformer fusing EEG DE features and peripheral signals.

    Config keys:
        n_classes, n_channels, n_bands, n_peripheral_feat, d_model, nhead,
        n_layers, lr, batch_size, n_epochs, device.
    """

    def __init__(
        self,
        n_classes: int = 3,
        n_channels: int = 62,
        n_bands: int = 5,
        n_peripheral_feat: int = 7,
        d_model: int = 64,
        nhead: int = 4,
        n_layers: int = 2,
        lr: float = 1e-3,
        batch_size: int = 64,
        n_epochs: int = 100,
        dropout: float = 0.1,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(n_classes=n_classes, device=device)
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.n_peripheral_feat = n_peripheral_feat
        self.d_model = d_model
        self.nhead = nhead
        self.n_layers_tf = n_layers
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dropout = dropout
        self.seed = seed
        self._build()

    def _build(self) -> None:
        self.network = _TransformerMM(
            n_classes=self.n_classes,
            n_channels=self.n_channels,
            n_bands=self.n_bands,
            n_peripheral_feat=self.n_peripheral_feat,
            d_model=self.d_model,
            nhead=self.nhead,
            n_layers=self.n_layers_tf,
            dropout=self.dropout,
        ).to(self.device)

    def _prepare_data(
        self, X: np.ndarray | dict[str, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Split input into EEG and peripheral tensors.

        If *X* is a dict it should have ``'eeg'`` and optionally ``'peripheral'`` keys.
        If it is a single array, the first ``n_channels * n_bands`` cols are
        EEG and the remainder peripheral (or None if exact match).
        """
        if isinstance(X, dict):
            eeg = torch.as_tensor(X["eeg"], dtype=torch.float32).to(self.device)
            if "peripheral" in X:
                periph = torch.as_tensor(X["peripheral"], dtype=torch.float32).to(
                    self.device
                )
            else:
                periph = None
        else:
            X_t = torch.as_tensor(X, dtype=torch.float32).to(self.device)
            eeg_dim = self.n_channels * self.n_bands
            eeg = X_t[:, :eeg_dim].reshape(-1, self.n_channels, self.n_bands)
            remaining = X_t[:, eeg_dim:]
            periph = remaining if remaining.shape[1] > 0 else None
        if eeg.ndim == 2:
            eeg = eeg.reshape(-1, self.n_channels, self.n_bands)
        return eeg, periph

    def fit(
        self,
        X_train: np.ndarray | dict[str, np.ndarray],
        y_train: np.ndarray,
        X_val: np.ndarray | dict[str, np.ndarray] | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Train the multimodal Transformer.

        Args:
            X_train: Dict with ``'eeg'`` ``(N, C, bands)`` and ``'peripheral'``
                ``(N, feat_dim)`` keys, or a flat array.
            y_train: Labels.
            X_val: Optional validation data.
            y_val: Optional validation labels.

        Returns:
            Training history dict.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        net = self.network
        net.to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        loss_fn = nn.CrossEntropyLoss()
        stopper = EarlyStopping(patience=10, min_delta=1e-4, mode="min")

        eeg_train, periph_train = self._prepare_data(X_train)
        y_t = torch.as_tensor(y_train, dtype=torch.long).to(self.device)
        train_ds = TensorDataset(eeg_train, periph_train, y_t)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            eeg_val, periph_val = self._prepare_data(X_val)
            y_v = torch.as_tensor(y_val, dtype=torch.long).to(self.device)
            val_ds = TensorDataset(eeg_val, periph_val, y_v)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        history: dict[str, list[float]] = {"train_loss": [], "val_acc": []}

        for epoch in tqdm(range(self.n_epochs), desc="Transformer-MM", leave=False):
            net.train()
            total_loss = 0.0
            n_batches = 0
            for eeg_b, periph_b, yb in train_loader:
                optimizer.zero_grad()
                logits = net(eeg_b, periph_b)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            history["train_loss"].append(avg_loss)
            scheduler.step(avg_loss)

            if val_loader is not None:
                net.eval()
                correct = total = 0
                with torch.no_grad():
                    for eeg_b, periph_b, yb in val_loader:
                        preds = net(eeg_b, periph_b).argmax(dim=-1)
                        correct += (preds == yb).sum().item()
                        total += yb.size(0)
                history["val_acc"].append(correct / max(total, 1))

            if stopper(avg_loss):
                logger.info("Transformer-MM early stopping at epoch %d", epoch + 1)
                break

        return history

    @torch.no_grad()
    def forward_features(
        self, X: np.ndarray | dict[str, np.ndarray]
    ) -> torch.Tensor:
        """Return CLS embedding before the classification head.

        Args:
            X: Input features (dict or array).

        Returns:
            CLS embedding ``(N, d_model)``.
        """
        self.network.eval()
        eeg, periph = self._prepare_data(X)
        return self.network.forward_features(eeg, periph)

    @torch.no_grad()
    def predict(self, X: np.ndarray | dict[str, np.ndarray]) -> np.ndarray:
        """Return class predictions.

        Args:
            X: Input features.

        Returns:
            Predicted class indices.
        """
        self.network.eval()
        eeg, periph = self._prepare_data(X)
        return self.network(eeg, periph).argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(
        self, X: np.ndarray | dict[str, np.ndarray]
    ) -> np.ndarray:
        """Return softmax probabilities.

        Args:
            X: Input features.

        Returns:
            Probabilities ``(N, n_classes)``.
        """
        self.network.eval()
        eeg, periph = self._prepare_data(X)
        logits = self.network(eeg, periph)
        return torch.softmax(logits, dim=-1).cpu().numpy()
