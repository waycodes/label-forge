"""
Ray Data determinism configuration.

Provides switches for ordering guarantees and deterministic execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import ray.data


class RayDeterminismMode(str, Enum):
    """Ray Data determinism mode."""

    # Strict ordering - preserve_order=True
    STRICT = "strict"

    # Standard - no ordering guarantees but keyed by row_id
    STANDARD = "standard"

    # Performance - maximize throughput, no ordering
    PERFORMANCE = "performance"


@dataclass
class RayDeterminismConfig:
    """Configuration for Ray Data determinism."""

    mode: RayDeterminismMode = RayDeterminismMode.STANDARD
    preserve_order: bool = False
    locality_with_output: bool = False

    @classmethod
    def from_mode(cls, mode: RayDeterminismMode) -> RayDeterminismConfig:
        """
        Create config from mode.

        Args:
            mode: Determinism mode.

        Returns:
            Configured RayDeterminismConfig.
        """
        if mode == RayDeterminismMode.STRICT:
            return cls(
                mode=mode,
                preserve_order=True,
                locality_with_output=True,
            )
        elif mode == RayDeterminismMode.PERFORMANCE:
            return cls(
                mode=mode,
                preserve_order=False,
                locality_with_output=False,
            )
        else:  # STANDARD
            return cls(
                mode=mode,
                preserve_order=False,
                locality_with_output=False,
            )

    @property
    def description(self) -> str:
        """Get human-readable description."""
        if self.mode == RayDeterminismMode.STRICT:
            return (
                "Strict mode: preserve_order=True for deterministic ordering. "
                "May reduce throughput but ensures reproducible row order."
            )
        elif self.mode == RayDeterminismMode.PERFORMANCE:
            return (
                "Performance mode: no ordering guarantees, maximum throughput. "
                "Row order may vary between runs."
            )
        else:
            return (
                "Standard mode: no ordering guarantees but all outputs keyed by row_id. "
                "Good balance of reproducibility and performance."
            )


def set_deterministic_mode(
    mode: RayDeterminismMode = RayDeterminismMode.STANDARD,
) -> RayDeterminismConfig:
    """
    Configure Ray Data for deterministic execution.

    Args:
        mode: Determinism mode to set.

    Returns:
        Applied configuration.

    Example:
        >>> config = set_deterministic_mode(RayDeterminismMode.STRICT)
        >>> print(config.preserve_order)
        True
    """
    import ray.data

    config = RayDeterminismConfig.from_mode(mode)

    # Get current data context
    ctx = ray.data.DataContext.get_current()

    # Apply settings
    if config.preserve_order:
        ctx.execution_options.preserve_order = True

    if config.locality_with_output:
        ctx.execution_options.locality_with_output = True

    return config


def get_current_ray_determinism() -> RayDeterminismConfig:
    """
    Get current Ray Data determinism configuration.

    Returns:
        Current configuration.
    """
    import ray.data

    ctx = ray.data.DataContext.get_current()
    preserve_order = getattr(ctx.execution_options, "preserve_order", False)

    if preserve_order:
        return RayDeterminismConfig(
            mode=RayDeterminismMode.STRICT,
            preserve_order=True,
        )
    else:
        return RayDeterminismConfig(
            mode=RayDeterminismMode.STANDARD,
            preserve_order=False,
        )


def create_execution_options(
    preserve_order: bool = False,
    resource_limits: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Create execution options dict for Ray Data operations.

    Args:
        preserve_order: Whether to preserve row order.
        resource_limits: Optional resource limits (CPU, GPU).

    Returns:
        Dict of execution options.
    """
    options: dict[str, Any] = {}

    if preserve_order:
        options["preserve_order"] = True

    if resource_limits:
        options["resource_limits"] = resource_limits

    return options


def apply_determinism_to_context(
    config: RayDeterminismConfig,
) -> None:
    """
    Apply determinism configuration to current Ray Data context.

    Args:
        config: Configuration to apply.
    """
    import ray.data

    ctx = ray.data.DataContext.get_current()

    ctx.execution_options.preserve_order = config.preserve_order
    ctx.execution_options.locality_with_output = config.locality_with_output


# Warnings for determinism
RAY_DETERMINISM_WARNINGS = [
    "Ray Data iteration order is not guaranteed unless preserve_order=True.",
    "With preserve_order=True, throughput may be reduced due to ordering constraints.",
    "LabelForge uses row_id for correctness - ordering is for reproducibility only.",
    "For maximum reproducibility, also enable vLLM determinism settings.",
]
