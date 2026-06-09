# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""YAML-based experiment configuration with pydantic v2 validation."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from emokit.utils import EmoKitConfigError

logger = logging.getLogger(__name__)


class ExperimentConfig(BaseModel):
    """Top-level experiment metadata."""

    name: str
    seed: int = 42
    device: str = "cpu"


class DatasetConfig(BaseModel):
    """Dataset specification."""

    name: str
    root: str | None = "data/"
    subjects: list[int] | None = None
    window_sec: float | None = 4.0
    overlap: float | None = 0.5
    modalities: list[str] | None = None
    label_axis: str | None = None
    params: dict[str, Any] | None = None

    @field_validator("overlap")
    @classmethod
    def _check_overlap(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v < 1.0):
            raise ValueError(f"overlap must be in [0, 1), got {v}")
        return v


class FeatureStepConfig(BaseModel):
    """Single feature extraction step."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class FeaturePipelineConfig(BaseModel):
    """Ordered list of feature extraction steps."""

    steps: list[FeatureStepConfig] = Field(default_factory=list)


class ModelConfig(BaseModel):
    """Model specification."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class MultiModelSpec(BaseModel):
    """Single model entry for batch evaluation configs."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class EvaluationConfig(BaseModel):
    """Evaluation protocol settings."""

    protocol: str = "loso"
    val_fraction: float = 0.1

    @field_validator("protocol")
    @classmethod
    def _check_protocol(cls, v: str) -> str:
        allowed = {"loso", "subject_dependent", "session", "cross_corpus"}
        if v not in allowed:
            raise ValueError(f"protocol must be one of {sorted(allowed)}, got '{v}'")
        return v

    @field_validator("val_fraction")
    @classmethod
    def _check_val_fraction(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError(f"val_fraction must be in [0, 1), got {v}")
        return v


class OutputConfig(BaseModel):
    """Output and checkpoint settings."""

    results_dir: str = "results/"
    save_checkpoints: bool = False


class FullConfig(BaseModel):
    """Complete experiment configuration aggregating all sub-configs."""

    model_config = {"protected_namespaces": ()}

    experiment: ExperimentConfig
    dataset: DatasetConfig
    target_dataset: DatasetConfig | None = None
    feature_pipeline: FeaturePipelineConfig = Field(
        default_factory=lambda: FeaturePipelineConfig(steps=[])
    )
    model: ModelConfig | None = None
    model_defaults: dict[str, Any] = Field(default_factory=dict)
    models_to_run: list[MultiModelSpec] = Field(default_factory=list)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("models_to_run")
    @classmethod
    def _ensure_model_present(
        cls, v: list[MultiModelSpec], info: Any
    ) -> list[MultiModelSpec]:
        data = info.data
        if data.get("model") is None and not v:
            raise ValueError("Either 'model' or 'models_to_run' must be provided.")
        return v


class ConfigLoader:
    """Load and validate experiment YAML files into :class:`FullConfig`.

    Example::

        cfg = ConfigLoader.load("configs/deap_loso_dgcnn.yaml")
        print(cfg.experiment.name)
    """

    @classmethod
    def load(cls, yaml_path: str) -> FullConfig:
        """Parse a YAML file and return a validated :class:`FullConfig`.

        Args:
            yaml_path: Filesystem path to the YAML configuration file.

        Returns:
            Validated :class:`FullConfig` instance.

        Raises:
            EmoKitConfigError: If the file cannot be read or validation fails.
        """
        path = Path(yaml_path)
        if not path.exists():
            raise EmoKitConfigError(f"Config file not found: {path}")

        try:
            data = _load_yaml_with_base(path)
        except yaml.YAMLError as exc:
            raise EmoKitConfigError(f"Invalid YAML in {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise EmoKitConfigError(
                f"Expected a YAML mapping at the top level, got {type(data).__name__}"
            )

        try:
            return FullConfig(**_expand_env(data))
        except ValidationError as exc:
            readable = _format_validation_errors(exc)
            raise EmoKitConfigError(
                f"Config validation failed for {path}:\n{readable}"
            ) from exc


def _format_validation_errors(exc: ValidationError) -> str:
    """Produce human-readable lines from pydantic validation errors."""
    lines: list[str] = []
    for err in exc.errors():
        loc = " -> ".join(str(x) for x in err["loc"])
        lines.append(f"  [{loc}] {err['msg']}")
    return "\n".join(lines)


def _load_yaml_with_base(path: Path) -> dict[str, Any]:
    """Load YAML with optional ``_base_`` inheritance."""
    raw = path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        return data

    base_name = data.pop("_base_", None)
    if not base_name:
        return data

    base_path = (path.parent / base_name).resolve()
    if not base_path.exists():
        raise EmoKitConfigError(f"Base config not found: {base_path}")

    base_data = _load_yaml_with_base(base_path)
    if not isinstance(base_data, dict):
        raise EmoKitConfigError(
            f"Base config must be a YAML mapping, got {type(base_data).__name__}"
        )
    return _deep_merge(base_data, data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_env(obj: Any) -> Any:
    """Recursively expand ``${VAR}`` references in string values."""
    if isinstance(obj, str):
        return re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)),
            obj,
        )
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj
