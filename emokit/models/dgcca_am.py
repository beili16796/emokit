# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Deep Generalised CCA with Attention Mechanism (DGCCA-AM) for tri-modal fusion."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from emokit.models.base import BaseModel, EarlyStopping, registry

logger = logging.getLogger(__name__)


class _ModalityEncoder(nn.Module):
    """Two-layer MLP encoder for a single modality.

    Args:
        n_feat: Input feature dimension.
        hidden_dim: Output / hidden dimension.
    """

    def __init__(self, n_feat: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 256),
            nn.ReLU(),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DGCCAAM(nn.Module):
    """Internal DGCCA-AM network with three modality encoders and attention fusion.

    Args:
        n_feat_eeg: EEG feature dimension.
        n_feat_gsr: GSR feature dimension.
        n_feat_ecg: ECG feature dimension.
        hidden_dim: Common encoding dimension.
        n_classes: Number of output classes.
    """

    def __init__(
        self,
        n_feat_eeg: int,
        n_feat_gsr: int,
        n_feat_ecg: int,
        hidden_dim: int = 128,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.enc_eeg = _ModalityEncoder(n_feat_eeg, hidden_dim)
        self.enc_gsr = _ModalityEncoder(n_feat_gsr, hidden_dim)
        self.enc_ecg = _ModalityEncoder(n_feat_ecg, hidden_dim)

        self.attn_W = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)

        self.classifier = nn.Linear(hidden_dim, n_classes)

    def _attention(
        self, hiddens: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted sum of modality embeddings.

        Args:
            hiddens: List of tensors each ``(B, hidden_dim)``.

        Returns:
            Tuple of (fused ``(B, hidden_dim)``, weights ``(B, n_modalities)``).
        """
        stacked = torch.stack(hiddens, dim=1)  # (B, M, H)
        scores = self.attn_v(torch.tanh(self.attn_W(stacked)))  # (B, M, 1)
        alpha = torch.softmax(scores, dim=1)  # (B, M, 1)
        fused = (alpha * stacked).sum(dim=1)  # (B, H)
        return fused, alpha.squeeze(-1)

    def forward(
        self,
        x_eeg: torch.Tensor,
        x_gsr: torch.Tensor,
        x_ecg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x_eeg: EEG features ``(B, n_feat_eeg)``.
            x_gsr: GSR features ``(B, n_feat_gsr)``.
            x_ecg: ECG features ``(B, n_feat_ecg)``.

        Returns:
            Tuple of (logits ``(B, n_classes)``, attention weights ``(B, 3)``).
        """
        h_eeg = self.enc_eeg(x_eeg)
        h_gsr = self.enc_gsr(x_gsr)
        h_ecg = self.enc_ecg(x_ecg)
        fused, alpha = self._attention([h_eeg, h_gsr, h_ecg])
        logits = self.classifier(fused)
        return logits, alpha


@registry.register("DGCCA-AM")
class DGCCAAMModel(BaseModel):
    """Deep Generalised CCA with Attention Mechanism.

    Three modality encoders (EEG, GSR, ECG) fused via learned attention.

    Input: ``X_train`` is dict ``{'eeg': arr, 'gsr': arr, 'ecg': arr}``.

    Config keys:
        n_classes, n_feat_eeg, n_feat_gsr, n_feat_ecg, hidden_dim, lr,
        batch_size, n_epochs, device.
    """

    def __init__(
        self,
        n_classes: int = 3,
        n_feat_eeg: int = 310,
        n_feat_gsr: int = 32,
        n_feat_ecg: int = 16,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        batch_size: int = 64,
        n_epochs: int = 100,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(n_classes=n_classes, device=device)
        self.n_feat_eeg = n_feat_eeg
        self.n_feat_gsr = n_feat_gsr
        self.n_feat_ecg = n_feat_ecg
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.seed = seed
        self._build()

    def _build(self) -> None:
        self.network = _DGCCAAM(
            n_feat_eeg=self.n_feat_eeg,
            n_feat_gsr=self.n_feat_gsr,
            n_feat_ecg=self.n_feat_ecg,
            hidden_dim=self.hidden_dim,
            n_classes=self.n_classes,
        ).to(self.device)

    def _prepare(
        self, X: dict[str, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eeg = torch.as_tensor(X["eeg"], dtype=torch.float32).to(self.device)
        gsr = torch.as_tensor(X["gsr"], dtype=torch.float32).to(self.device)
        ecg = torch.as_tensor(X["ecg"], dtype=torch.float32).to(self.device)
        return eeg, gsr, ecg

    def fit(
        self,
        X_train: dict[str, np.ndarray],
        y_train: np.ndarray,
        X_val: dict[str, np.ndarray] | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Train the DGCCA-AM model.

        Args:
            X_train: Dict with ``'eeg'``, ``'gsr'``, ``'ecg'`` arrays.
            y_train: Labels.
            X_val: Optional validation dict.
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

        eeg_t, gsr_t, ecg_t = self._prepare(X_train)
        y_t = torch.as_tensor(y_train, dtype=torch.long).to(self.device)
        train_ds = TensorDataset(eeg_t, gsr_t, ecg_t, y_t)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            eeg_v, gsr_v, ecg_v = self._prepare(X_val)
            y_v = torch.as_tensor(y_val, dtype=torch.long).to(self.device)
            val_ds = TensorDataset(eeg_v, gsr_v, ecg_v, y_v)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        history: dict[str, list[float]] = {"train_loss": [], "val_acc": []}

        for epoch in tqdm(range(self.n_epochs), desc="DGCCA-AM", leave=False):
            net.train()
            total_loss = 0.0
            n_batches = 0
            for eeg_b, gsr_b, ecg_b, yb in train_loader:
                optimizer.zero_grad()
                logits, _ = net(eeg_b, gsr_b, ecg_b)
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
                    for eeg_b, gsr_b, ecg_b, yb in val_loader:
                        logits, _ = net(eeg_b, gsr_b, ecg_b)
                        preds = logits.argmax(dim=-1)
                        correct += (preds == yb).sum().item()
                        total += yb.size(0)
                history["val_acc"].append(correct / max(total, 1))

            if stopper(avg_loss):
                logger.info("DGCCA-AM early stopping at epoch %d", epoch + 1)
                break

        return history

    @torch.no_grad()
    def predict(self, X: dict[str, np.ndarray]) -> np.ndarray:
        """Return class predictions.

        Args:
            X: Dict with ``'eeg'``, ``'gsr'``, ``'ecg'`` arrays.

        Returns:
            Predicted class indices.
        """
        self.network.eval()
        eeg, gsr, ecg = self._prepare(X)
        logits, _ = self.network(eeg, gsr, ecg)
        return logits.argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, X: dict[str, np.ndarray]) -> np.ndarray:
        """Return softmax probabilities.

        Args:
            X: Dict with ``'eeg'``, ``'gsr'``, ``'ecg'`` arrays.

        Returns:
            Probabilities ``(N, n_classes)``.
        """
        self.network.eval()
        eeg, gsr, ecg = self._prepare(X)
        logits, _ = self.network(eeg, gsr, ecg)
        return torch.softmax(logits, dim=-1).cpu().numpy()

    @torch.no_grad()
    def get_attention_weights(self, X: dict[str, np.ndarray]) -> np.ndarray:
        """Return per-sample attention weights across the three modalities.

        Args:
            X: Dict with ``'eeg'``, ``'gsr'``, ``'ecg'`` arrays.

        Returns:
            Attention weights ``(N, 3)`` summing to 1 per sample.
        """
        self.network.eval()
        eeg, gsr, ecg = self._prepare(X)
        _, alpha = self.network(eeg, gsr, ecg)
        return alpha.cpu().numpy()
