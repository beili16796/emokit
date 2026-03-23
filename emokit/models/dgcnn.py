# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Dynamical Graph Convolutional Neural Network (DGCNN) — Chebyshev spectral variant."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from emokit.models.base import BaseModel, EarlyStopping, registry
from emokit.utils import EmoKitModelError

logger = logging.getLogger(__name__)


class ChebGraphConv(nn.Module):
    """Chebyshev spectral graph convolution layer (Song et al. 2020, Eq.11-12).

    Computes h_k = T_k(L_tilde) @ X using the recurrence
    T_0=I, T_1=L_tilde, T_k = 2*L_tilde*T_{k-1} - T_{k-2}.

    Args:
        in_features: Input feature dimension per node.
        out_features: Output feature dimension per node.
        K: Chebyshev polynomial order.
    """

    def __init__(self, in_features: int, out_features: int, K: int = 2) -> None:
        super().__init__()
        self.K = K
        self.theta = nn.Parameter(torch.empty(K, in_features, out_features))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, x: torch.Tensor, L_tilde: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features ``(batch, N_nodes, in_features)``.
            L_tilde: Scaled normalized Laplacian ``(N_nodes, N_nodes)``.

        Returns:
            Updated features ``(batch, N_nodes, out_features)``.
        """
        Tx_0 = x
        out = torch.matmul(Tx_0, self.theta[0])

        if self.K > 1:
            Tx_1 = torch.einsum("ij,bjk->bik", L_tilde, x)
            out = out + torch.matmul(Tx_1, self.theta[1])

            Tx_prev, Tx_cur = Tx_0, Tx_1
            for k in range(2, self.K):
                Tx_next = 2.0 * torch.einsum("ij,bjk->bik", L_tilde, Tx_cur) - Tx_prev
                out = out + torch.matmul(Tx_next, self.theta[k])
                Tx_prev, Tx_cur = Tx_cur, Tx_next

        return F.relu(out)


class _DGCNN(nn.Module):
    """Internal DGCNN network with learnable adjacency and Chebyshev convolution.

    Args:
        n_classes: Number of output classes.
        n_channels: Number of EEG channels (graph nodes).
        n_bands: Number of frequency-band features per node.
        hidden_dim: Hidden dimension for graph conv layers.
        K: Chebyshev polynomial order.
    """

    def __init__(
        self,
        n_classes: int,
        n_channels: int = 62,
        n_bands: int = 5,
        hidden_dim: int = 64,
        K: int = 2,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels

        self.adjacency = nn.Parameter(
            torch.empty(n_channels, n_channels).uniform_(0, 0.5)
        )

        self.gc1 = ChebGraphConv(n_bands, hidden_dim, K=K)
        self.gc2 = ChebGraphConv(hidden_dim, hidden_dim, K=K)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Sequential(
            nn.Linear(n_channels * hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )

    def get_adjacency(self) -> torch.Tensor:
        """Return the symmetrised, non-negative adjacency matrix."""
        A_sym = (self.adjacency + self.adjacency.t()) / 2.0
        return F.relu(A_sym)

    def _normalized_laplacian(self) -> torch.Tensor:
        """Compute the scaled normalised Laplacian L_tilde = L_norm - I.

        Approximates 2*L/lambda_max - I with lambda_max ≈ 2.
        """
        A = self.get_adjacency()
        D = A.sum(dim=1).clamp(min=1e-8)
        D_inv_sqrt = D.pow(-0.5)
        L_norm = torch.eye(self.n_channels, device=A.device) - (
            D_inv_sqrt.unsqueeze(1) * A * D_inv_sqrt.unsqueeze(0)
        )
        return L_norm - torch.eye(self.n_channels, device=A.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: DE features ``(batch, C, n_bands)``.

        Returns:
            Logits ``(batch, n_classes)``.
        """
        assert x.ndim == 3, f"Expected (B, C, bands), got {x.shape}"
        L = self._normalized_laplacian()

        h = self.gc1(x, L)
        h = self.bn1(h.transpose(1, 2)).transpose(1, 2)
        h = h + self.gc2(h, L)
        h = self.bn2(h.transpose(1, 2)).transpose(1, 2)
        h = self.dropout(h)

        h = h.reshape(h.size(0), -1)
        return self.fc(h)


@registry.register("DGCNN")
class DGCNNModel(BaseModel):
    """Dynamical Graph CNN for EEG emotion recognition (Song et al. 2020).

    Uses Chebyshev spectral graph convolution with a learnable adjacency
    matrix.  L2 regularisation is applied via the optimizer's
    ``weight_decay`` parameter, consistent with the original paper.

    Config keys:
        n_classes, n_channels, n_bands, hidden_dim, K, lr, weight_decay,
        batch_size, n_epochs, device.
    """

    def __init__(
        self,
        n_classes: int = 3,
        n_channels: int = 62,
        n_bands: int = 5,
        hidden_dim: int = 64,
        K: int = 2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        n_epochs: int = 100,
        device: str = "cpu",
        seed: int | None = None,
        lambda_reg: float | None = None,
    ) -> None:
        super().__init__(n_classes=n_classes, device=device)
        self.n_channels = n_channels
        self.n_bands = n_bands
        self.hidden_dim = hidden_dim
        self.K = K
        self.lr = lr
        self.weight_decay = lambda_reg if lambda_reg is not None else weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.seed = seed
        self._build()

    @property
    def A(self) -> nn.Parameter:
        """Alias for the learnable adjacency parameter."""
        return self.network.adjacency

    def _build(self) -> None:
        self.network = _DGCNN(
            n_classes=self.n_classes,
            n_channels=self.n_channels,
            n_bands=self.n_bands,
            hidden_dim=self.hidden_dim,
            K=self.K,
        ).to(self.device)

    def configure_optimizer(self) -> torch.optim.Optimizer:
        """Return Adam optimizer with L2 weight decay (lambda=1e-4)."""
        return torch.optim.Adam(
            self.network.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Train the DGCNN model.

        Args:
            X_train: DE features ``(N, C, bands)``.
            y_train: Labels.
            X_val: Optional validation features.
            y_val: Optional validation labels.

        Returns:
            Training history dict.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        net = self.network
        net.to(self.device)
        net.train()

        optimizer = torch.optim.Adam(
            net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        loss_fn = nn.CrossEntropyLoss()
        stopper = EarlyStopping(patience=10, min_delta=1e-4, mode="min")

        train_ds = TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            val_ds = TensorDataset(
                torch.as_tensor(X_val, dtype=torch.float32),
                torch.as_tensor(y_val, dtype=torch.long),
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        history: dict[str, list[float]] = {"train_loss": [], "val_acc": []}

        for epoch in tqdm(range(self.n_epochs), desc="DGCNN Training", leave=False):
            net.train()
            total_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            history["train_loss"].append(avg_loss)
            scheduler.step(avg_loss)

            val_acc = 0.0
            if val_loader is not None:
                net.eval()
                correct = total = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        preds = net(xb).argmax(dim=-1)
                        correct += (preds == yb).sum().item()
                        total += yb.size(0)
                val_acc = correct / max(total, 1)
                history["val_acc"].append(val_acc)

            if stopper(avg_loss):
                logger.info("DGCNN early stopping at epoch %d", epoch + 1)
                break

        return history

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(X, dtype=torch.float32).to(self.device)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions."""
        self.network.eval()
        return self.network(self._to_tensor(X)).argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities."""
        self.network.eval()
        logits = self.network(self._to_tensor(X))
        return torch.softmax(logits, dim=-1).cpu().numpy()

    @torch.no_grad()
    def get_adjacency_matrix(self) -> np.ndarray:
        """Return the learned symmetric adjacency matrix as NumPy array.

        Returns:
            Array of shape ``(n_channels, n_channels)``.
        """
        if self.network is None:
            raise EmoKitModelError("Network not initialised.")
        return self.network.get_adjacency().detach().cpu().numpy()
