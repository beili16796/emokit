# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Prototype-Representation / Pairwise-Loss (PR-PL) model for cross-subject
EEG emotion recognition."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from emokit.models.base import BaseModel, EarlyStopping, registry

logger = logging.getLogger(__name__)


class _GradRevFn(torch.autograd.Function):
    """Gradient Reversal Layer (standard GRL)."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class _PRPL(nn.Module):
    """Internal PR-PL network with domain adversarial adaptation for LOSO.

    Args:
        n_feat: Input feature dimension.
        n_classes: Number of classes.
        prototype_dim: Dimension of each prototype vector.
        margin: Pairwise loss margin.
        lambda_adv: Weight for domain adversarial loss.
    """

    def __init__(
        self,
        n_feat: int,
        n_classes: int = 3,
        prototype_dim: int = 128,
        margin: float = 0.5,
        lambda_adv: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.margin = margin
        self.prototype_dim = prototype_dim
        self.lambda_adv = lambda_adv

        self.encoder = nn.Sequential(
            nn.Linear(n_feat, 256),
            nn.ReLU(),
            nn.Linear(256, prototype_dim),
            nn.ReLU(),
        )

        self.register_buffer("prototypes", torch.randn(n_classes, prototype_dim) * 0.02)
        self.register_buffer("_proto_counts", torch.zeros(n_classes, dtype=torch.long))

        self.classifier = nn.Linear(prototype_dim, n_classes)

        self.domain_disc = nn.Sequential(
            nn.Linear(prototype_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input features to the prototype space.

        Args:
            x: Input ``(B, n_feat)``.

        Returns:
            Encoded features ``(B, prototype_dim)``.
        """
        return self.encoder(x)

    def update_prototypes(self, z: torch.Tensor, y: torch.Tensor) -> None:
        """Update per-class prototypes as running mean during training.

        Args:
            z: Encoded features ``(B, prototype_dim)``.
            y: Class labels ``(B,)``.
        """
        for c in range(self.n_classes):
            mask = y == c
            if mask.any():
                z_c = z[mask].detach().mean(dim=0)
                count = self._proto_counts[c].item()
                if count == 0:
                    self.prototypes[c] = z_c
                else:
                    denom = count + 1
                    momentum = 1.0 / denom
                    self.prototypes[c] = (1 - momentum) * self.prototypes[
                        c
                    ] + momentum * z_c
                self._proto_counts[c] += mask.sum()

    def pairwise_loss(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the pairwise prototype loss.

        For each sample: max(0, margin + d(z, p_correct) - d(z, p_negative)).

        Args:
            z: Encoded features ``(B, prototype_dim)``.
            y: Class labels ``(B,)``.

        Returns:
            Scalar pairwise loss.
        """
        protos = self.prototypes.unsqueeze(0).expand(z.size(0), -1, -1)  # (B, C, dim)
        d_all = torch.cdist(z.unsqueeze(1), protos).squeeze(1)  # (B, n_classes)

        d_pos = d_all.gather(1, y.unsqueeze(1)).squeeze(1)  # (B,)

        mask = torch.ones_like(d_all, dtype=torch.bool)
        mask.scatter_(1, y.unsqueeze(1), False)
        d_neg_all = d_all.masked_fill(~mask, float("inf"))
        d_neg = d_neg_all.min(dim=1).values

        loss = F.relu(self.margin + d_pos - d_neg)
        return loss.mean()

    def pairwise_loss_pseudo(self, z: torch.Tensor) -> torch.Tensor:
        """Pairwise loss for target samples using pseudo-labels (nearest prototype).

        Args:
            z: Encoded target features ``(B, prototype_dim)``.

        Returns:
            Scalar pairwise loss with pseudo-labels.
        """
        protos = self.prototypes.unsqueeze(0).expand(z.size(0), -1, -1)
        d_all = torch.cdist(z.unsqueeze(1), protos).squeeze(1)
        pseudo_y = d_all.argmin(dim=1)
        return self.pairwise_loss(z, pseudo_y)

    def domain_adversarial_loss(
        self, z_source: torch.Tensor, z_target: torch.Tensor, alpha: float = 1.0
    ) -> torch.Tensor:
        """Domain adversarial loss via GRL.

        Args:
            z_source: Source domain embeddings ``(B_s, prototype_dim)``.
            z_target: Target domain embeddings ``(B_t, prototype_dim)``.
            alpha: GRL scaling factor.

        Returns:
            Scalar adversarial loss.
        """
        z_s_rev = _GradRevFn.apply(z_source, alpha)
        z_t_rev = _GradRevFn.apply(z_target, alpha)

        pred_s = self.domain_disc(z_s_rev)
        pred_t = self.domain_disc(z_t_rev)

        label_s = torch.zeros(
            z_source.size(0), dtype=torch.long, device=z_source.device
        )
        label_t = torch.ones(
            z_target.size(0), dtype=torch.long, device=z_target.device
        )

        loss = F.cross_entropy(pred_s, label_s) + F.cross_entropy(pred_t, label_t)
        return loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classification forward pass.

        Args:
            x: Input ``(B, n_feat)``.

        Returns:
            Logits ``(B, n_classes)``.
        """
        z = self.encode(x)
        return self.classifier(z)


@registry.register("PR-PL")
class PRPLModel(BaseModel):
    """Prototype-Representation / Pairwise-Loss model with domain adversarial
    adaptation.

    Learns per-class prototypes (updated as running mean) in a shared embedding
    space with a contrastive pairwise loss.  Supports cross-subject LOSO via
    optional domain adversarial training when ``X_target`` is provided to ``fit()``.

    Config keys:
        n_classes, n_feat, prototype_dim, margin, lr, batch_size, n_epochs,
        lambda_pair, lambda_adv, device.
    """

    def __init__(
        self,
        n_classes: int = 3,
        n_feat: int = 310,
        prototype_dim: int = 128,
        margin: float = 0.5,
        lr: float = 1e-3,
        batch_size: int = 64,
        n_epochs: int = 100,
        lambda_pair: float = 0.5,
        lambda_adv: float = 0.1,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        super().__init__(n_classes=n_classes, device=device)
        self.n_feat = n_feat
        self.prototype_dim = prototype_dim
        self.margin = margin
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lambda_pair = lambda_pair
        self.lambda_adv = lambda_adv
        self.seed = seed
        self._build()

    def _build(self) -> None:
        self.network = _PRPL(
            n_feat=self.n_feat,
            n_classes=self.n_classes,
            prototype_dim=self.prototype_dim,
            margin=self.margin,
            lambda_adv=self.lambda_adv,
        ).to(self.device)

    def get_attention_weights(self) -> None:
        """PR-PL is unimodal — no attention weights."""
        return None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        subject_ids: np.ndarray | None = None,
        X_target: np.ndarray | None = None,
        n_epochs: int | None = None,
    ) -> dict[str, list[float]]:
        """Train the PR-PL model.

        Args:
            X_train: Features ``(N, n_feat)``.
            y_train: Labels ``(N,)``.
            X_val: Optional validation features.
            y_val: Optional validation labels.
            subject_ids: Ignored (kept for API compatibility).
            X_target: Optional unlabelled target-domain features for domain
                adversarial adaptation (LOSO).

        Returns:
            Training history dict.
        """
        actual_epochs = n_epochs if n_epochs is not None else self.n_epochs

        if self.seed is not None:
            torch.manual_seed(self.seed)

        net = self.network
        net.to(self.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        ce_loss = nn.CrossEntropyLoss()
        stopper = EarlyStopping(patience=10, min_delta=1e-4, mode="min")

        X_t = torch.as_tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.as_tensor(y_train, dtype=torch.long).to(self.device)
        train_ds = TensorDataset(X_t, y_t)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        target_loader = None
        if X_target is not None:
            X_tgt = torch.as_tensor(X_target, dtype=torch.float32).to(self.device)
            target_ds = TensorDataset(X_tgt)
            target_loader = DataLoader(
                target_ds, batch_size=self.batch_size, shuffle=True
            )

        val_loader = None
        if X_val is not None and y_val is not None:
            X_v = torch.as_tensor(X_val, dtype=torch.float32).to(self.device)
            y_v = torch.as_tensor(y_val, dtype=torch.long).to(self.device)
            val_ds = TensorDataset(X_v, y_v)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        history: dict[str, list[float]] = {"train_loss": [], "val_acc": []}

        for epoch in tqdm(range(actual_epochs), desc="PR-PL Training", leave=False):
            net.train()
            total_loss = 0.0
            n_batches = 0

            target_iter = iter(target_loader) if target_loader is not None else None

            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = net(xb)
                z = net.encode(xb)
                net.update_prototypes(z, yb)

                loss = ce_loss(logits, yb) + self.lambda_pair * net.pairwise_loss(z, yb)

                if target_iter is not None:
                    try:
                        x_tgt_b = next(target_iter)[0]
                    except StopIteration:
                        target_iter = iter(target_loader)
                        x_tgt_b = next(target_iter)[0]

                    z_tgt = net.encode(x_tgt_b)
                    adv_progress = epoch / max(actual_epochs - 1, 1)
                    alpha = 2.0 / (1.0 + np.exp(-10.0 * adv_progress)) - 1.0
                    loss = loss + self.lambda_adv * net.domain_adversarial_loss(
                        z, z_tgt, alpha
                    )
                    loss = loss + self.lambda_pair * net.pairwise_loss_pseudo(z_tgt)

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
                    for xb, yb in val_loader:
                        preds = net(xb).argmax(dim=-1)
                        correct += (preds == yb).sum().item()
                        total += yb.size(0)
                history["val_acc"].append(correct / max(total, 1))

            if stopper(avg_loss):
                logger.info("PR-PL early stopping at epoch %d", epoch + 1)
                break

        return history

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(X, dtype=torch.float32).to(self.device)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions.

        Args:
            X: Features ``(N, n_feat)``.

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
            X: Features ``(N, n_feat)``.

        Returns:
            Probabilities ``(N, n_classes)``.
        """
        self.network.eval()
        logits = self.network(self._to_tensor(X))
        return torch.softmax(logits, dim=-1).cpu().numpy()
