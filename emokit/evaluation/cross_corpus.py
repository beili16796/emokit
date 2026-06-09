# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Cross-corpus evaluation: train on one dataset, test on another."""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np

from emokit.datasets.base import BaseDataset
from emokit.evaluation.protocols import (
    _build_pipeline_from_config,
    _clone_pipeline,
    _prepare_model_features,
    _stratified_val_split_any,
    compute_metrics,
)
from emokit.features.base import FeaturePipeline
from emokit.models.base import build_model
from emokit.utils import EmoKitConfigError, set_seed

logger = logging.getLogger(__name__)

# Standard 10-20 system channel names (superset for matching)
_STANDARD_10_20 = {
    "FP1", "FP2", "FPZ",
    "AF3", "AF4", "AFZ", "AF7", "AF8",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "FZ",
    "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FCZ", "FT7", "FT8",
    "C1", "C2", "C3", "C4", "C5", "C6", "CZ",
    "T7", "T8", "TP7", "TP8",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPZ",
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "PZ",
    "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", "POZ",
    "O1", "O2", "OZ",
}


def find_common_channels(
    source_channels: list[str],
    target_channels: list[str],
) -> tuple[list[int], list[int], list[str]]:
    """Find shared EEG channels between two datasets by 10-20 name matching.

    Returns:
        Tuple of (source_indices, target_indices, common_channel_names).
    """
    src_lookup: dict[str, int] = {}
    for i, name in enumerate(source_channels):
        src_lookup.setdefault(name.upper(), i)

    common_names: list[str] = []
    src_idx: list[int] = []
    tgt_idx: list[int] = []

    # Iterate in target order so both corpora are aligned to the same montage.
    # Duplicate channel labels in a source corpus map to the first occurrence.
    for j, t_ch in enumerate(ch.upper() for ch in target_channels):
        if t_ch in src_lookup and t_ch not in {name.upper() for name in common_names}:
            src_idx.append(src_lookup[t_ch])
            tgt_idx.append(j)
            common_names.append(target_channels[j])

    return src_idx, tgt_idx, common_names


def _select_channels(
    data: dict[str, np.ndarray],
    channel_indices: list[int],
) -> dict[str, np.ndarray]:
    """Select a subset of EEG channels from raw data dict."""
    result = dict(data)
    if "eeg" in result and result["eeg"] is not None and channel_indices:
        eeg = result["eeg"]
        # Raw/windowed EEG is usually (n_trials, n_channels, n_samples).
        # Some lightweight/pre-extracted fixtures use (n_trials, 1, n_channels);
        # support both without treating the trial axis as channels.
        if eeg.ndim == 3:
            max_idx = max(channel_indices)
            if max_idx < eeg.shape[1]:
                result["eeg"] = eeg[:, channel_indices, :]
            elif max_idx < eeg.shape[2]:
                result["eeg"] = eeg[:, :, channel_indices]
        elif eeg.ndim == 2 and max(channel_indices) < eeg.shape[1]:
            result["eeg"] = eeg[:, channel_indices]
    return result


class CrossCorpusEvaluator:
    """Cross-corpus evaluator: train on source dataset, test on target dataset.

    Uses channel subset matching based on 10-20 system electrode names
    to align different-montage datasets (e.g. SEED 62ch → DREAMER 14ch).

    Args:
        source_dataset: Dataset used for training.
        target_dataset: Dataset used for testing.
        feature_pipeline: Feature extraction pipeline.
        model_config: Model constructor arguments.
        model_name: Registered model name.
        seed: Random seed.
        val_fraction: Fraction of source data for validation.
    """

    def __init__(
        self,
        source_dataset: BaseDataset,
        target_dataset: BaseDataset,
        feature_pipeline: FeaturePipeline,
        model_config: dict[str, Any],
        model_name: str,
        seed: int = 42,
        val_fraction: float = 0.1,
    ) -> None:
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.feature_pipeline = feature_pipeline
        self.model_config = model_config
        self.model_name = model_name
        self.seed = seed
        self.val_fraction = val_fraction

        src_ch = source_dataset.get_channel_names("eeg")
        tgt_ch = target_dataset.get_channel_names("eeg")
        self.src_idx, self.tgt_idx, self.common_channels = find_common_channels(
            src_ch, tgt_ch
        )

        if not self.common_channels:
            raise EmoKitConfigError(
                f"No common EEG channels between "
                f"{type(source_dataset).__name__} and "
                f"{type(target_dataset).__name__}"
            )

        logger.info(
            "Cross-corpus channel alignment: %d common channels out of "
            "source=%d, target=%d — %s",
            len(self.common_channels),
            len(src_ch),
            len(tgt_ch),
            self.common_channels,
        )

    def run(self) -> dict[str, Any]:
        """Execute cross-corpus evaluation.

        Trains on all source subjects, tests on each target subject
        independently (LOSO-style reporting on target).

        Returns:
            Dict with ``per_subject`` (target subjects), ``mean``, ``std``,
            ``config``, and ``common_channels``.
        """
        set_seed(self.seed)

        # --- Load source data (channel-subset) ---
        source_raws: list[dict[str, np.ndarray]] = []
        source_labels: list[np.ndarray] = []
        for sid in self.source_dataset.get_subject_ids():
            raw = self.source_dataset.read_raw(sid)
            raw = _select_channels(raw, self.src_idx)
            source_raws.append(raw)
            source_labels.append(np.asarray(raw["labels"]))

        y_source = np.concatenate(source_labels, axis=0)

        # --- Evaluate on each target subject ---
        target_sids = self.target_dataset.get_subject_ids()
        per_subject: dict[int, dict[str, Any]] = {}

        for test_sid in target_sids:
            logger.info("Cross-corpus test: target subject %d", test_sid)

            test_raw = self.target_dataset.read_raw(test_sid)
            test_raw = _select_channels(test_raw, self.tgt_idx)
            y_test = np.asarray(test_raw["labels"])

            pipeline = _clone_pipeline(self.feature_pipeline)
            X_train, X_test = _prepare_model_features(
                train_raws=source_raws,
                test_raw=test_raw,
                y_train=y_source,
                pipeline=pipeline,
                model_name=self.model_name,
                model_config=self.model_config,
            )

            X_tr, y_tr, X_val, y_val = _stratified_val_split_any(
                X_train, y_source, self.val_fraction, self.seed
            )

            model = build_model(self.model_name, self.model_config)
            model.fit(X_tr, y_tr, X_val, y_val)

            y_pred = model.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            per_subject[test_sid] = metrics
            logger.info(
                "Target subject %d — acc=%.4f  f1=%.4f",
                test_sid,
                metrics["accuracy"],
                metrics["f1_macro"],
            )

        metric_keys = [k for k in next(iter(per_subject.values())) if k != "confusion_matrix"]
        values = {k: [m[k] for m in per_subject.values()] for k in metric_keys}

        return {
            "per_subject": per_subject,
            "mean": {k: float(np.mean(v)) for k, v in values.items()},
            "std": {k: float(np.std(v)) for k, v in values.items()},
            "common_channels": self.common_channels,
            "n_common_channels": len(self.common_channels),
            "config": {
                "source_dataset": type(self.source_dataset).__name__,
                "target_dataset": type(self.target_dataset).__name__,
                "model_name": self.model_name,
                "seed": self.seed,
                "protocol": "cross_corpus",
                "channel_alignment": len(self.common_channels)
                != len(self.source_dataset.get_channel_names("eeg"))
                or len(self.common_channels)
                != len(self.target_dataset.get_channel_names("eeg")),
                "source_channels": len(self.source_dataset.get_channel_names("eeg")),
                "target_channels": len(self.target_dataset.get_channel_names("eeg")),
            },
        }
