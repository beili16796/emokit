# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""YAML-based experiment configuration with pydantic v2 validation."""

from __future__ import annotations

import logging
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


class EvaluationConfig(BaseModel):
    """Evaluation protocol settings."""

    protocol: str = "loso"
    val_fraction: float = 0.1

    @field_validator("protocol")
    @classmethod
    def _check_protocol(cls, v: str) -> str:
        allowed = {"loso", "subject_dependent", "session"}
        if v not in allowed:
            raise ValueError(
                f"protocol must be one of {sorted(allowed)}, got '{v}'"
            )
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

    experiment: ExperimentConfig
    dataset: DatasetConfig
    feature_pipeline: FeaturePipelineConfig = Field(
        default_factory=lambda: FeaturePipelineConfig(steps=[])
    )
    model: ModelConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


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
            raw = path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise EmoKitConfigError(f"Invalid YAML in {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise EmoKitConfigError(
                f"Expected a YAML mapping at the top level, got {type(data).__name__}"
            )

        try:
            return FullConfig(**data)
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
