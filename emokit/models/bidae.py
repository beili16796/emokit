# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Bi-modal Denoising Autoencoder (BiDAE) with shared bottleneck for emotion
recognition."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from emokit.models.base import BaseModel, EarlyStopping, registry

logger = logging.getLogger(__name__)


class _MLP(nn.Module):
    """Simple feed-forward MLP with ReLU activations.

    Args:
        dims: List of layer dimensions, e.g. ``[512, 256, 128]``.
    """

    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _BiDAE(nn.Module):
    """Internal BiDAE network with two autoencoders sharing a bottleneck.

    Args:
        n_feat1: Dimensionality of modality-1 input.
        n_feat2: Dimensionality of modality-2 input.
        bottleneck_dim: Shared bottleneck dimension.
        n_classes: Number of output classes.
    """

    def __init__(
        self,
        n_feat1: int,
        n_feat2: int,
        bottleneck_dim: int = 128,
        n_classes: int = 3,
    ) -> None:
        super().__init__()
        self.encoder1 = _MLP([n_feat1, 512, 256, bottleneck_dim])
        self.decoder1 = _MLP([bottleneck_dim, 256, 512, n_feat1])
        self.encoder2 = nn.Sequential(
            nn.Linear(n_feat2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck_dim),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_feat2),
        )
        self.classifier = nn.Linear(bottleneck_dim, n_classes)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x1: Modality-1 features ``(B, n_feat1)``.
            x2: Modality-2 features ``(B, n_feat2)``.

        Returns:
            Tuple of (logits, z1, z2, x1_recon, x2_recon).
        """
        z1 = self.encoder1(x1)
        z2 = self.encoder2(x2)
        x1_recon = self.decoder1(z1)
        x2_recon = self.decoder2(z2)
        z = (z1 + z2) / 2
        logits = self.classifier(z)
        return logits, z1, z2, x1_recon, x2_recon


@registry.register("BiDAE")
class BiDAEModel(BaseModel):
    """Bi-modal Denoising Autoencoder with classification head.

    Engineering note: Original paper uses RBM with Contrastive Divergence
    pretraining.  This implementation uses standard AutoEncoder with MSE
    reconstruction loss, which is the modern engineering equivalent.

    Loss = CE + lambda * (recon1 + recon2) + mu * MSE(z1, z2).

    Input: ``X_train`` is a dict ``{'mod1': arr1, 'mod2': arr2}``.

    Config keys:
        n_classes, n_feat1, n_feat2, bottleneck_dim, lambda_recon, mu_align,
        lr, batch_size, n_epochs, device.
    """

    multimodal = True

    def __init__(
        self,
        n_classes: int = 3,
        n_feat1: int | None = None,
        n_feat2: int | None = None,
        n_feat_mod1: int = 310,
        n_feat_mod2: int = 32,
        bottleneck_dim: int = 128,
        lambda_recon: float = 0.1,
        lambda_rec: float | None = None,
        mu_align: float | None = None,
        lambda_align: float = 0.01,
        lr: float = 1e-3,
        batch_size: int = 64,
        n_epochs: int = 100,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(n_classes=n_classes, device=device)
        self.n_feat1 = n_feat1 if n_feat1 is not None else n_feat_mod1
        self.n_feat2 = n_feat2 if n_feat2 is not None else n_feat_mod2
        self.bottleneck_dim = bottleneck_dim
        self.lambda_recon = lambda_rec if lambda_rec is not None else lambda_recon
        self.mu_align = mu_align if mu_align is not None else lambda_align
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.seed = seed
        self._build()

    def _build(self) -> None:
        self.network = _BiDAE(
            n_feat1=self.n_feat1,
            n_feat2=self.n_feat2,
            bottleneck_dim=self.bottleneck_dim,
            n_classes=self.n_classes,
        ).to(self.device)

    def _prepare(
        self, X: dict[str, np.ndarray] | np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(X, dict):
            if "mod1" in X and "mod2" in X:
                x1 = torch.as_tensor(X["mod1"], dtype=torch.float32).to(self.device)
                x2 = torch.as_tensor(X["mod2"], dtype=torch.float32).to(self.device)
            else:
                keys = sorted(X.keys())
                if len(keys) < 2:
                    raise ValueError(
                        f"BiDAE requires 2 modalities, got {len(keys)}: {keys}"
                    )
                x1 = torch.as_tensor(X[keys[0]], dtype=torch.float32).to(self.device)
                x2 = torch.as_tensor(X[keys[1]], dtype=torch.float32).to(self.device)
        elif isinstance(X, torch.Tensor):
            x1 = X.to(self.device)
            x2 = torch.zeros(X.shape[0], self.n_feat2, device=self.device)
        else:
            x1 = torch.as_tensor(X, dtype=torch.float32).to(self.device)
            x2 = torch.zeros(X.shape[0], self.n_feat2, device=self.device)
        return x1, x2

    def compute_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the full BiDAE loss (CE + recon + alignment).

        Args:
            x1: Modality-1 features ``(B, n_feat1)``.
            x2: Modality-2 features ``(B, n_feat2)``.
            y: Class labels ``(B,)``.

        Returns:
            Scalar loss tensor.
        """
        logits, z1, z2, x1_recon, x2_recon = self.network(x1, x2)
        ce = nn.functional.cross_entropy(logits, y)
        mse = nn.functional.mse_loss
        return (
            ce
            + self.lambda_recon * (mse(x1_recon, x1) + mse(x2_recon, x2))
            + self.mu_align * mse(z1, z2)
        )

    def fit(
        self,
        X_train: dict[str, np.ndarray] | np.ndarray,
        y_train: np.ndarray,
        X_val: dict[str, np.ndarray] | np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Train the BiDAE model.

        Args:
            X_train: Dict with ``'mod1'`` and ``'mod2'`` arrays.
            y_train: Labels.
            X_val: Optional validation data dict.
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
        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()
        stopper = EarlyStopping(patience=10, min_delta=1e-4, mode="min")

        x1_train, x2_train = self._prepare(X_train)
        y_t = torch.as_tensor(y_train, dtype=torch.long).to(self.device)
        train_ds = TensorDataset(x1_train, x2_train, y_t)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            x1_val, x2_val = self._prepare(X_val)
            y_v = torch.as_tensor(y_val, dtype=torch.long).to(self.device)
            val_ds = TensorDataset(x1_val, x2_val, y_v)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        history: dict[str, list[float]] = {"train_loss": [], "val_acc": []}

        for epoch in tqdm(range(self.n_epochs), desc="BiDAE Training", leave=False):
            net.train()
            total_loss = 0.0
            n_batches = 0
            for x1b, x2b, yb in train_loader:
                optimizer.zero_grad()
                logits, z1, z2, x1_recon, x2_recon = net(x1b, x2b)
                recon = mse_loss(x1_recon, x1b) + mse_loss(x2_recon, x2b)
                loss = (
                    ce_loss(logits, yb)
                    + self.lambda_recon * recon
                    + self.mu_align * mse_loss(z1, z2)
                )
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
                    for x1b, x2b, yb in val_loader:
                        logits = net(x1b, x2b)[0]
                        preds = logits.argmax(dim=-1)
                        correct += (preds == yb).sum().item()
                        total += yb.size(0)
                history["val_acc"].append(correct / max(total, 1))

            if stopper(avg_loss):
                logger.info("BiDAE early stopping at epoch %d", epoch + 1)
                break

        return history

    @torch.no_grad()
    def predict(self, X: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
        """Return class predictions.

        Args:
            X: Dict with ``'mod1'`` and ``'mod2'`` arrays.

        Returns:
            Predicted class indices.
        """
        self.network.eval()
        x1, x2 = self._prepare(X)
        logits = self.network(x1, x2)[0]
        return logits.argmax(dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict_proba(self, X: dict[str, np.ndarray] | np.ndarray) -> np.ndarray:
        """Return softmax probabilities.

        Args:
            X: Dict with ``'mod1'`` and ``'mod2'`` arrays.

        Returns:
            Probabilities ``(N, n_classes)``.
        """
        self.network.eval()
        x1, x2 = self._prepare(X)
        logits = self.network(x1, x2)[0]
        return torch.softmax(logits, dim=-1).cpu().numpy()
