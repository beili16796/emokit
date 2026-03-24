# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Base model abstractions, registry, early stopping, and standard training loop."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from emokit.utils import EmoKitModelError

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Dict-based registry mapping string names to model classes.

    Example::

        registry = ModelRegistry()

        @registry.register('my-model')
        class MyModel(BaseModel):
            ...

        cls = registry['my-model']
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseModel]] = {}

    def register(self, name: str):
        """Decorator that registers a model class under *name*.

        Args:
            name: Unique string identifier for the model.

        Returns:
            Decorator function.
        """

        def decorator(cls: type[BaseModel]) -> type[BaseModel]:
            if name in self._registry:
                logger.warning("Overwriting existing model '%s' in registry.", name)
            self._registry[name] = cls
            cls.registry_name = name
            return cls

        return decorator

    def __getitem__(self, name: str) -> type[BaseModel]:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise EmoKitModelError(
                f"Model '{name}' not found. Available: [{available}]"
            )
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def keys(self):
        return self._registry.keys()

    def items(self):
        return self._registry.items()


registry = ModelRegistry()


def build_model(name: str, config: dict[str, Any]) -> BaseModel:
    """Convenience factory that looks up *name* in the global registry.

    Args:
        name: Registered model name.
        config: Configuration dict forwarded to ``from_config``.

    Returns:
        Instantiated model.
    """
    cls = registry[name]
    return cls.from_config(config)


class BaseModel(ABC):
    """Abstract base class for all EmoKit models.

    Every concrete model must implement ``_build_network``, ``fit``,
    ``predict``, and ``predict_proba``.
    """

    registry_name: str = ""
    multimodal: bool = False

    def __init__(self, n_classes: int, device: str = "cpu") -> None:
        self.n_classes = n_classes
        self.device = torch.device(device)
        self.network: nn.Module | None = None

    @abstractmethod
    def fit(
        self,
        X_train: Any,
        y_train: np.ndarray,
        X_val: Any | None = None,
        y_val: np.ndarray | None = None,
    ) -> dict[str, list[float]]:
        """Train the model.

        Args:
            X_train: Training features (array or dict of arrays).
            y_train: Training labels.
            X_val: Optional validation features.
            y_val: Optional validation labels.

        Returns:
            History dict with keys like ``train_loss`` and ``val_acc``.
        """

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Return integer class predictions.

        Args:
            X: Input features.

        Returns:
            1-D array of predicted class indices.
        """

    @abstractmethod
    def predict_proba(self, X: Any) -> np.ndarray:
        """Return softmax probability predictions.

        Args:
            X: Input features.

        Returns:
            2-D array of shape ``(N, n_classes)``.
        """

    def evaluate(self, X: Any, y: np.ndarray) -> dict[str, float]:
        """Compute standard classification metrics.

        Args:
            X: Input features.
            y: Ground-truth labels.

        Returns:
            Dict with ``accuracy``, ``f1_macro``, ``f1_weighted``.
        """
        preds = self.predict(X)
        return {
            "accuracy": float(accuracy_score(y, preds)),
            "f1_macro": float(f1_score(y, preds, average="macro", zero_division=0)),
            "f1_weighted": float(
                f1_score(y, preds, average="weighted", zero_division=0)
            ),
        }

    def save(self, path: str) -> None:
        """Persist model weights to disk.

        Args:
            path: File path for the checkpoint.
        """
        if self.network is None:
            raise EmoKitModelError("No network to save — call fit() first.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.network.state_dict(), path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load model weights from disk.

        Args:
            path: File path of the checkpoint.
        """
        if self.network is None:
            raise EmoKitModelError("Build the network before loading weights.")
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(state)
        self.network.to(self.device)
        logger.info("Model loaded from %s", path)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> BaseModel:
        """Construct model from a config dict.

        Args:
            config: Keyword arguments forwarded to ``__init__``.

        Returns:
            Model instance.
        """
        return cls(**config)


class EarlyStopping:
    """Monitors a metric and signals when training should stop.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: ``'min'`` for loss-like metrics, ``'max'`` for accuracy-like.
    """

    def __init__(
        self, patience: int = 10, min_delta: float = 0.001, mode: str = "min"
    ) -> None:
        assert mode in ("min", "max"), f"mode must be 'min' or 'max', got '{mode}'"
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.counter: int = 0

    def __call__(self, metric: float) -> bool:
        """Update with new metric value.

        Args:
            metric: Current epoch metric.

        Returns:
            ``True`` if training should stop.
        """
        if self.best is None:
            self.best = metric
            return False

        improved = (
            (metric < self.best - self.min_delta)
            if self.mode == "min"
            else (metric > self.best + self.min_delta)
        )

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def reset(self) -> None:
        """Reset internal state."""
        self.best = None
        self.counter = 0


class StandardTrainer:
    """Wraps a PyTorch training loop with LR scheduling, gradient clipping,
    and early stopping.

    Args:
        n_epochs: Maximum training epochs.
        lr: Learning rate.
        batch_size: Mini-batch size.
        max_grad_norm: If > 0, clip gradient norms.
        patience: Early stopping patience (0 to disable).
        device: Torch device string.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 64,
        max_grad_norm: float = 0.0,
        patience: int = 10,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.patience = patience
        self.device = torch.device(device)
        self.seed = seed

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, shuffle: bool = True
    ) -> DataLoader:
        """Create a ``DataLoader`` from numpy arrays."""
        X_t = torch.as_tensor(X, dtype=torch.float32)
        y_t = torch.as_tensor(y, dtype=torch.long)
        return DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def train(
        self,
        model: nn.Module,
        train_loader: DataLoader | None = None,
        val_loader: DataLoader | None = None,
        X_train: np.ndarray | None = None,
        y_train: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        loss_fn: nn.Module | None = None,
        extra_loss_fn: Any | None = None,
    ) -> dict[str, list[float]]:
        """Run the training loop.

        Either pass pre-built ``DataLoader`` objects **or** raw numpy arrays
        (``X_train``/``y_train``).  If both are provided the loaders take
        precedence.

        Args:
            model: PyTorch ``nn.Module`` to train.
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            X_train: Training features as numpy array.
            y_train: Training labels as numpy array.
            X_val: Validation features as numpy array.
            y_val: Validation labels as numpy array.
            loss_fn: Loss function (defaults to ``CrossEntropyLoss``).
            extra_loss_fn: Optional callable ``(model) -> Tensor`` that returns
                an additional loss term (e.g. regularisation).

        Returns:
            History dict with ``train_loss`` and optionally ``val_acc``.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)

        if train_loader is None:
            assert X_train is not None and y_train is not None
            train_loader = self._make_loader(X_train, y_train, shuffle=True)
        if val_loader is None and X_val is not None and y_val is not None:
            val_loader = self._make_loader(X_val, y_val, shuffle=False)

        model.to(self.device)
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        stopper = (
            EarlyStopping(patience=self.patience, min_delta=1e-4, mode="min")
            if self.patience > 0
            else None
        )

        history: dict[str, list[float]] = {"train_loss": [], "val_acc": []}

        for epoch in tqdm(range(self.n_epochs), desc="Training", leave=False):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                xb, yb = batch[0].to(self.device), batch[1].to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                if extra_loss_fn is not None:
                    loss = loss + extra_loss_fn(model)
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_loss)
            scheduler.step(avg_loss)

            val_acc = 0.0
            if val_loader is not None:
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch in val_loader:
                        xb, yb = batch[0].to(self.device), batch[1].to(self.device)
                        preds = model(xb).argmax(dim=-1)
                        correct += (preds == yb).sum().item()
                        total += yb.size(0)
                val_acc = correct / max(total, 1)
                history["val_acc"].append(val_acc)

            logger.debug(
                "Epoch %d/%d  loss=%.4f  val_acc=%.4f",
                epoch + 1,
                self.n_epochs,
                avg_loss,
                val_acc,
            )

            if stopper is not None and stopper(avg_loss):
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        return history
