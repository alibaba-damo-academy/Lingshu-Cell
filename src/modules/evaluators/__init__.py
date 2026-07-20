"""Evaluation metric modules for the eval workflow.

Registry maps metric names to their module classes. The eval workflow
uses this to instantiate only the requested metrics.
"""

from __future__ import annotations

from typing import Any, Type, Union

from .base import ExternalEvalModule, MetricModule

METRIC_REGISTRY: dict[str, Type[Union[MetricModule, ExternalEvalModule]]] = {}


def register_metric(name: str):
    """Decorator to register a metric module class."""
    def wrapper(cls):
        METRIC_REGISTRY[name] = cls
        return cls
    return wrapper


def get_metric(name: str) -> Type[Union[MetricModule, ExternalEvalModule]]:
    """Look up a metric module class by name.

    Args:
        name: Registered metric name.

    Returns:
        The metric module class.

    Raises:
        KeyError: If the name is not registered.
    """
    if name not in METRIC_REGISTRY:
        raise KeyError(
            f"Unknown metric '{name}'. Available: {list(METRIC_REGISTRY.keys())}"
        )
    return METRIC_REGISTRY[name]


__all__ = [
    "ExternalEvalModule",
    "MetricModule",
    "METRIC_REGISTRY",
    "get_metric",
    "register_metric",
]
