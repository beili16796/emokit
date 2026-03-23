# MIT License
# Copyright (c) 2024 EmoKit Contributors
# See LICENSE for full text.

"""Base classes for feature transforms and pipeline orchestration."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import yaml

from emokit.utils import EmoKitFeatureError

logger = logging.getLogger(__name__)


class TransformRegistry:
    """Dict-based registry mapping string names to transform classes.

    Example::

        registry = TransformRegistry()

        @registry.register("MyTransform")
        class MyTransform(BaseTransform):
            ...

        cls = registry["MyTransform"]
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[BaseTransform]] = {}

    def register(self, name: str) -> Any:
        """Return a decorator that registers a transform class under *name*.

        Args:
            name: Unique string identifier for the transform.

        Returns:
            Decorator that registers the class and returns it unchanged.

        Raises:
            EmoKitFeatureError: If *name* is already registered.
        """

        def decorator(cls: type[BaseTransform]) -> type[BaseTransform]:
            if name in self._registry:
                raise EmoKitFeatureError(
                    f"Transform '{name}' is already registered."
                )
            self._registry[name] = cls
            logger.debug("Registered transform '%s' -> %s", name, cls.__name__)
            return cls

        return decorator

    def __getitem__(self, name: str) -> type[BaseTransform]:
        """Look up a registered transform class by name.

        Args:
            name: Previously registered identifier.

        Returns:
            The transform class.

        Raises:
            EmoKitFeatureError: If *name* is not found.
        """
        if name not in self._registry:
            raise EmoKitFeatureError(
                f"Transform '{name}' not found in registry. "
                f"Available: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def keys(self) -> list[str]:
        """Return all registered transform names."""
        return list(self._registry.keys())


GLOBAL_REGISTRY = TransformRegistry()
"""Module-level registry shared by all built-in transforms."""


class BaseTransform(ABC):
    """Abstract base class for all feature transforms.

    Subclasses must implement :meth:`fit` and :meth:`transform`.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> BaseTransform:
        """Fit the transform on training data.

        Args:
            X: Input array.
            y: Optional labels.

        Returns:
            self
        """

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the transform.

        Args:
            X: Input array.

        Returns:
            Transformed array.
        """

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        """Fit and then transform *X*.

        Args:
            X: Input array.
            y: Optional labels.

        Returns:
            Transformed array.
        """
        return self.fit(X, y).transform(X)


class FeaturePipeline:
    """Sequential pipeline of named :class:`BaseTransform` steps.

    Args:
        steps: Ordered list of ``(name, transform)`` pairs.

    Raises:
        EmoKitFeatureError: If *steps* contains duplicate names.
    """

    def __init__(self, steps: list[tuple[str, BaseTransform]]) -> None:
        names = [n for n, _ in steps]
        if len(names) != len(set(names)):
            raise EmoKitFeatureError("Pipeline step names must be unique.")
        self.steps = steps

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> FeaturePipeline:
        """Fit each step sequentially, passing transformed output forward.

        Args:
            X: Input array.
            y: Optional labels (forwarded to each step).

        Returns:
            self
        """
        for name, step in self.steps:
            logger.debug("Fitting step '%s'", name)
            X = step.fit_transform(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform *X* through each fitted step sequentially.

        Args:
            X: Input array.

        Returns:
            Transformed array.
        """
        for name, step in self.steps:
            logger.debug("Transforming step '%s'", name)
            X = step.transform(X)
        return X

    def fit_transform(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> np.ndarray:
        """Fit the pipeline and return the final transformed output.

        Args:
            X: Input array.
            y: Optional labels.

        Returns:
            Transformed array.
        """
        self.fit(X, y)
        return self.transform(X)

    # -- Serialization --------------------------------------------------------

    def to_yaml(self) -> str:
        """Serialize the pipeline configuration to a YAML string.

        Returns:
            YAML representation of the pipeline.
        """
        config: list[dict[str, Any]] = []
        for name, step in self.steps:
            entry: dict[str, Any] = {
                "name": name,
                "transform": type(step).__name__,
            }
            params = {
                k: v
                for k, v in step.__dict__.items()
                if not k.startswith("_")
            }
            if params:
                entry["params"] = _convert_numpy(params)
            config.append(entry)
        return yaml.dump({"pipeline": config}, default_flow_style=False)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> FeaturePipeline:
        """Deserialize a pipeline from a YAML string.

        Args:
            yaml_str: YAML produced by :meth:`to_yaml`.

        Returns:
            Reconstructed :class:`FeaturePipeline`.

        Raises:
            EmoKitFeatureError: If the YAML is malformed or references
                unknown transforms.
        """
        try:
            data = yaml.safe_load(yaml_str)
        except yaml.YAMLError as exc:
            raise EmoKitFeatureError(f"Invalid YAML: {exc}") from exc

        if not isinstance(data, dict) or "pipeline" not in data:
            raise EmoKitFeatureError(
                "YAML must contain a top-level 'pipeline' key."
            )
        return cls.from_config(data)

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> FeaturePipeline:
        """Instantiate a pipeline from a config dictionary.

        The dictionary must have a ``"pipeline"`` key whose value is a list
        of dicts, each with ``"name"``, ``"transform"``, and optional
        ``"params"`` entries.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Constructed :class:`FeaturePipeline`.

        Raises:
            EmoKitFeatureError: On invalid config or unknown transform name.
        """
        if "pipeline" not in config_dict:
            raise EmoKitFeatureError(
                "Config dict must contain a 'pipeline' key."
            )

        steps: list[tuple[str, BaseTransform]] = []
        for entry in config_dict["pipeline"]:
            transform_name = entry["transform"]
            params = entry.get("params", {})
            try:
                transform_cls = GLOBAL_REGISTRY[transform_name]
            except EmoKitFeatureError:
                raise EmoKitFeatureError(
                    f"Unknown transform '{transform_name}' in config."
                )
            steps.append((entry["name"], transform_cls(**params)))
        return cls(steps)


def _convert_numpy(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for YAML safety."""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_numpy(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
