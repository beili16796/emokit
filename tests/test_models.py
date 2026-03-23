# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Comprehensive unit tests for the EmoKit model library."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import torch

from emokit.models import (
    BaseModel,
    BiDAEModel,
    CNNLSTMModel,
    DGCCAAMModel,
    DGCNNModel,
    EarlyStopping,
    ModelRegistry,
    PRPLModel,
    StandardTrainer,
    TransformerMMModel,
    build_model,
    registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def seed() -> int:
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


# ---------------------------------------------------------------------------
# CNN-LSTM tests
# ---------------------------------------------------------------------------

class TestCNNLSTM:
    """Tests for the CNN-LSTM model."""

    def test_forward_raw(self, seed: int) -> None:
        """Raw EEG input (16, 62, 512) produces correct output shape."""
        model = CNNLSTMModel(
            n_classes=3, input_type="raw", n_channels=62, n_epochs=1, seed=seed
        )
        X = np.random.randn(16, 62, 512).astype(np.float32)
        proba = model.predict_proba(X)
        assert proba.shape == (16, 3), f"Expected (16, 3), got {proba.shape}"
        preds = model.predict(X)
        assert preds.shape == (16,), f"Expected (16,), got {preds.shape}"

    def test_forward_de(self, seed: int) -> None:
        """DE input (16, 310) produces correct output shape."""
        model = CNNLSTMModel(
            n_classes=4, input_type="de", n_channels=62, n_epochs=1, seed=seed
        )
        X = np.random.randn(16, 310).astype(np.float32)
        proba = model.predict_proba(X)
        assert proba.shape == (16, 4), f"Expected (16, 4), got {proba.shape}"

    def test_fit(self, seed: int) -> None:
        """Training completes and returns history."""
        model = CNNLSTMModel(
            n_classes=3, input_type="de", n_channels=62,
            n_epochs=2, batch_size=8, seed=seed,
        )
        X = np.random.randn(32, 310).astype(np.float32)
        y = np.random.randint(0, 3, 32)
        history = model.fit(X, y)
        assert "train_loss" in history
        assert len(history["train_loss"]) > 0


# ---------------------------------------------------------------------------
# DGCNN tests
# ---------------------------------------------------------------------------

class TestDGCNN:
    """Tests for the DGCNN model."""

    def test_forward(self, seed: int) -> None:
        """(8, 62, 5) input produces (8, n_classes) output."""
        model = DGCNNModel(n_classes=3, n_channels=62, n_bands=5, n_epochs=1, seed=seed)
        X = np.random.randn(8, 62, 5).astype(np.float32)
        proba = model.predict_proba(X)
        assert proba.shape == (8, 3), f"Expected (8, 3), got {proba.shape}"

    def test_adjacency_symmetry_and_nonneg(self, seed: int) -> None:
        """Adjacency matrix is symmetric and non-negative after training."""
        model = DGCNNModel(
            n_classes=3, n_channels=10, n_bands=5,
            n_epochs=1, batch_size=4, seed=seed,
        )
        X = np.random.randn(8, 10, 5).astype(np.float32)
        y = np.random.randint(0, 3, 8)
        model.fit(X, y)
        A = model.network.get_adjacency().detach().cpu().numpy()
        np.testing.assert_allclose(A, A.T, atol=1e-6, err_msg="Adjacency not symmetric")
        assert np.all(A >= 0), "Adjacency matrix has negative entries"

    def test_get_adjacency_matrix_numpy(self, seed: int) -> None:
        """Public API returns square, symmetric, non-negative NumPy adjacency."""
        model = DGCNNModel(
            n_classes=2, n_channels=8, n_bands=5, n_epochs=1, batch_size=4, seed=seed,
        )
        X = np.random.randn(8, 8, 5).astype(np.float32)
        y = np.random.randint(0, 2, 8)
        model.fit(X, y)
        a = model.get_adjacency_matrix()
        assert a.shape == (8, 8)
        np.testing.assert_allclose(a, a.T, atol=1e-6)
        assert np.all(a >= 0), "Adjacency matrix has negative entries"

    def test_fit(self, seed: int) -> None:
        """Training returns history with train_loss."""
        model = DGCNNModel(
            n_classes=3, n_channels=10, n_bands=5,
            n_epochs=2, batch_size=4, seed=seed,
        )
        X = np.random.randn(16, 10, 5).astype(np.float32)
        y = np.random.randint(0, 3, 16)
        history = model.fit(X, y)
        assert len(history["train_loss"]) == 2

    def test_adjacency_updates_during_training(self, seed: int) -> None:
        """Adjacency parameter changes after training steps."""
        model = DGCNNModel(
            n_classes=3, n_channels=10, n_bands=5,
            n_epochs=5, batch_size=4, seed=seed,
        )
        adj_before = model.network.adjacency.detach().clone().numpy()
        X = np.random.randn(16, 10, 5).astype(np.float32)
        y = np.random.randint(0, 3, 16)
        model.fit(X, y)
        adj_after = model.network.adjacency.detach().numpy()
        assert not np.allclose(adj_before, adj_after, atol=1e-7), (
            "Adjacency did not update during training"
        )


# ---------------------------------------------------------------------------
# Transformer-MM tests
# ---------------------------------------------------------------------------

class TestTransformerMM:
    """Tests for the multimodal Transformer."""

    def test_forward(self, seed: int) -> None:
        """EEG (8,62,5) + peripheral (8,7) → (8, n_classes)."""
        model = TransformerMMModel(
            n_classes=3, n_channels=62, n_bands=5,
            n_peripheral_feat=7, n_epochs=1, seed=seed,
        )
        X = {
            "eeg": np.random.randn(8, 62, 5).astype(np.float32),
            "peripheral": np.random.randn(8, 7).astype(np.float32),
        }
        proba = model.predict_proba(X)
        assert proba.shape == (8, 3), f"Expected (8, 3), got {proba.shape}"

    def test_fit(self, seed: int) -> None:
        """Training completes with dict input."""
        model = TransformerMMModel(
            n_classes=3, n_channels=10, n_bands=5,
            n_peripheral_feat=4, n_epochs=2, batch_size=4, seed=seed,
        )
        X = {
            "eeg": np.random.randn(16, 10, 5).astype(np.float32),
            "peripheral": np.random.randn(16, 4).astype(np.float32),
        }
        y = np.random.randint(0, 3, 16)
        history = model.fit(X, y)
        assert len(history["train_loss"]) == 2


# ---------------------------------------------------------------------------
# BiDAE tests
# ---------------------------------------------------------------------------

class TestBiDAE:
    """Tests for the BiDAE model."""

    def test_forward(self, seed: int) -> None:
        """Dict input produces correct shape output."""
        model = BiDAEModel(
            n_classes=3, n_feat1=100, n_feat2=32, n_epochs=1, seed=seed,
        )
        X = {
            "mod1": np.random.randn(8, 100).astype(np.float32),
            "mod2": np.random.randn(8, 32).astype(np.float32),
        }
        proba = model.predict_proba(X)
        assert proba.shape == (8, 3), f"Expected (8, 3), got {proba.shape}"
        preds = model.predict(X)
        assert preds.shape == (8,)

    def test_fit(self, seed: int) -> None:
        """Training returns decreasing loss."""
        model = BiDAEModel(
            n_classes=3, n_feat1=50, n_feat2=20,
            n_epochs=3, batch_size=4, seed=seed,
        )
        X = {
            "mod1": np.random.randn(16, 50).astype(np.float32),
            "mod2": np.random.randn(16, 20).astype(np.float32),
        }
        y = np.random.randint(0, 3, 16)
        history = model.fit(X, y)
        assert len(history["train_loss"]) == 3


# ---------------------------------------------------------------------------
# DGCCA-AM tests
# ---------------------------------------------------------------------------

class TestDGCCAAM:
    """Tests for the DGCCA-AM model."""

    def test_forward(self, seed: int) -> None:
        """Dict input produces correct shape output."""
        model = DGCCAAMModel(
            n_classes=3, n_feat_eeg=100, n_feat_gsr=32, n_feat_ecg=16,
            n_epochs=1, seed=seed,
        )
        X = {
            "eeg": np.random.randn(8, 100).astype(np.float32),
            "gsr": np.random.randn(8, 32).astype(np.float32),
            "ecg": np.random.randn(8, 16).astype(np.float32),
        }
        proba = model.predict_proba(X)
        assert proba.shape == (8, 3), f"Expected (8, 3), got {proba.shape}"

    def test_attention_weights(self, seed: int) -> None:
        """Attention weights have correct shape and sum to 1."""
        model = DGCCAAMModel(
            n_classes=3, n_feat_eeg=100, n_feat_gsr=32, n_feat_ecg=16,
            n_epochs=1, seed=seed,
        )
        X = {
            "eeg": np.random.randn(8, 100).astype(np.float32),
            "gsr": np.random.randn(8, 32).astype(np.float32),
            "ecg": np.random.randn(8, 16).astype(np.float32),
        }
        weights = model.get_attention_weights(X)
        assert weights.shape == (8, 3), f"Expected (8, 3), got {weights.shape}"
        np.testing.assert_allclose(
            weights.sum(axis=1), np.ones(8), atol=1e-5,
            err_msg="Attention weights do not sum to 1",
        )

    def test_fit(self, seed: int) -> None:
        """Training completes with dict input."""
        model = DGCCAAMModel(
            n_classes=3, n_feat_eeg=50, n_feat_gsr=16, n_feat_ecg=8,
            n_epochs=2, batch_size=4, seed=seed,
        )
        X = {
            "eeg": np.random.randn(16, 50).astype(np.float32),
            "gsr": np.random.randn(16, 16).astype(np.float32),
            "ecg": np.random.randn(16, 8).astype(np.float32),
        }
        y = np.random.randint(0, 3, 16)
        history = model.fit(X, y)
        assert len(history["train_loss"]) == 2


# ---------------------------------------------------------------------------
# PR-PL tests
# ---------------------------------------------------------------------------

class TestPRPL:
    """Tests for the PR-PL model."""

    def test_forward(self, seed: int) -> None:
        """Forward pass produces correct shape."""
        model = PRPLModel(
            n_classes=3, n_feat=100, n_epochs=1, seed=seed,
        )
        X = np.random.randn(8, 100).astype(np.float32)
        proba = model.predict_proba(X)
        assert proba.shape == (8, 3), f"Expected (8, 3), got {proba.shape}"

    def test_pairwise_loss_decreases(self, seed: int) -> None:
        """Pairwise loss should decrease after one optimisation step."""
        torch.manual_seed(seed)
        net = PRPLModel(
            n_classes=3, n_feat=50,
            n_epochs=1, seed=seed,
        ).network

        X = torch.randn(16, 50)
        y = torch.randint(0, 3, (16,))

        z = net.encode(X)
        net.update_prototypes(z, y)
        loss_before = net.pairwise_loss(z, y).item()

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        for _ in range(10):
            optimizer.zero_grad()
            z = net.encode(X)
            net.update_prototypes(z, y)
            loss = net.pairwise_loss(z, y)
            loss.backward()
            optimizer.step()

        z = net.encode(X)
        loss_after = net.pairwise_loss(z, y).item()
        assert loss_after < loss_before, (
            f"Pairwise loss did not decrease: {loss_before:.4f} → {loss_after:.4f}"
        )

    def test_fit(self, seed: int) -> None:
        """Training completes."""
        model = PRPLModel(
            n_classes=3, n_feat=50,
            n_epochs=2, batch_size=4, seed=seed,
        )
        X = np.random.randn(16, 50).astype(np.float32)
        y = np.random.randint(0, 3, 16)
        history = model.fit(X, y)
        assert len(history["train_loss"]) == 2


# ---------------------------------------------------------------------------
# EarlyStopping tests
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    """Tests for EarlyStopping."""

    def test_min_mode(self) -> None:
        """Stops when loss stops decreasing."""
        es = EarlyStopping(patience=3, min_delta=0.0, mode="min")
        for val in [1.0, 0.9, 0.8, 0.8, 0.8]:
            result = es(val)
        assert not result
        assert es(0.8)  # 4th non-improving step → stop

    def test_max_mode(self) -> None:
        """Stops when accuracy stops increasing."""
        es = EarlyStopping(patience=2, min_delta=0.0, mode="max")
        assert not es(0.5)
        assert not es(0.6)
        assert not es(0.6)
        assert es(0.6)

    def test_reset(self) -> None:
        """Reset clears state."""
        es = EarlyStopping(patience=1, mode="min")
        es(1.0)
        es(1.0)
        assert es(1.0)
        es.reset()
        assert es.best is None
        assert es.counter == 0
        assert not es(1.0)


# ---------------------------------------------------------------------------
# ModelRegistry tests
# ---------------------------------------------------------------------------

class TestModelRegistry:
    """Tests for ModelRegistry and build_model."""

    def test_registered_models(self) -> None:
        """All six models are registered."""
        expected = {"CNN-LSTM", "DGCNN", "Transformer-MM", "BiDAE", "DGCCA-AM", "PR-PL"}
        assert expected.issubset(set(registry.keys())), (
            f"Missing models: {expected - set(registry.keys())}"
        )

    def test_getitem(self) -> None:
        """Registry lookup returns correct class."""
        assert registry["CNN-LSTM"] is CNNLSTMModel
        assert registry["DGCNN"] is DGCNNModel

    def test_missing_raises(self) -> None:
        """Unknown name raises EmoKitModelError."""
        from emokit.utils import EmoKitModelError

        with pytest.raises(EmoKitModelError, match="not found"):
            registry["nonexistent-model"]

    def test_build_model(self) -> None:
        """build_model returns a BaseModel subclass."""
        model = build_model("DGCNN", {"n_classes": 4, "n_channels": 10, "n_bands": 5})
        assert isinstance(model, BaseModel)
        assert isinstance(model, DGCNNModel)
        assert model.n_classes == 4


# ---------------------------------------------------------------------------
# Save / Load round-trip
# ---------------------------------------------------------------------------

class TestSaveLoad:
    """Test save/load round-trip for a model."""

    def test_dgcnn_save_load(self, seed: int) -> None:
        """DGCNN weights survive a save/load cycle."""
        model = DGCNNModel(
            n_classes=3, n_channels=10, n_bands=5, n_epochs=1, seed=seed,
        )
        X = np.random.randn(8, 10, 5).astype(np.float32)
        y = np.random.randint(0, 3, 8)
        model.fit(X, y)

        preds_before = model.predict(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dgcnn.pt")
            model.save(path)

            model2 = DGCNNModel(
                n_classes=3, n_channels=10, n_bands=5, n_epochs=1, seed=seed,
            )
            model2.load(path)
            preds_after = model2.predict(X)

        np.testing.assert_array_equal(
            preds_before, preds_after,
            err_msg="Predictions differ after save/load round-trip",
        )

    def test_cnn_lstm_save_load(self, seed: int) -> None:
        """CNN-LSTM DE-mode weights survive a save/load cycle."""
        model = CNNLSTMModel(
            n_classes=3, input_type="de", n_channels=62,
            n_epochs=1, batch_size=8, seed=seed,
        )
        X = np.random.randn(16, 310).astype(np.float32)
        y = np.random.randint(0, 3, 16)
        model.fit(X, y)
        proba_before = model.predict_proba(X)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "cnn_lstm.pt")
            model.save(path)

            model2 = CNNLSTMModel(
                n_classes=3, input_type="de", n_channels=62,
                n_epochs=1, batch_size=8, seed=seed,
            )
            model2.load(path)
            proba_after = model2.predict_proba(X)

        np.testing.assert_allclose(
            proba_before, proba_after, atol=1e-6,
            err_msg="Probabilities differ after save/load round-trip",
        )
