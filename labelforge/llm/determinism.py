"""
vLLM determinism toggles for reproducible inference.

Provides switches for batch invariance vs deterministic scheduling modes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


class DeterminismMode(str, Enum):
    """vLLM determinism mode."""

    # Batch-invariant: same outputs regardless of batching (recommended)
    BATCH_INVARIANT = "batch_invariant"

    # Deterministic scheduling: disable v1 multiprocessing (stricter)
    DETERMINISTIC_SCHEDULING = "deterministic_scheduling"

    # No determinism guarantees (best throughput)
    NONE = "none"


@dataclass
class DeterminismConfig:
    """Configuration for determinism mode."""

    mode: DeterminismMode
    seed: int | None = None

    @property
    def env_vars(self) -> dict[str, str]:
        """Get environment variables for this mode."""
        if self.mode == DeterminismMode.BATCH_INVARIANT:
            return {"VLLM_BATCH_INVARIANT": "1"}
        elif self.mode == DeterminismMode.DETERMINISTIC_SCHEDULING:
            return {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
        else:
            return {}

    @property
    def description(self) -> str:
        """Get human-readable description of this mode."""
        if self.mode == DeterminismMode.BATCH_INVARIANT:
            return (
                "Batch-invariant mode: outputs are independent of batching. "
                "Best balance of reproducibility and throughput."
            )
        elif self.mode == DeterminismMode.DETERMINISTIC_SCHEDULING:
            return (
                "Deterministic scheduling mode: disables v1 multiprocessing. "
                "Maximum reproducibility but may reduce throughput."
            )
        else:
            return "No determinism mode: best throughput, no reproducibility guarantees."


def apply_determinism_env_vars(mode: DeterminismMode) -> dict[str, str]:
    """
    Apply determinism environment variables.

    Args:
        mode: Determinism mode to apply.

    Returns:
        Dict of environment variables that were set.
    """
    config = DeterminismConfig(mode=mode)
    for key, value in config.env_vars.items():
        os.environ[key] = value
    return config.env_vars


def get_current_determinism_mode() -> DeterminismMode:
    """
    Detect the current determinism mode from environment.

    Returns:
        Current DeterminismMode based on env vars.
    """
    if os.environ.get("VLLM_BATCH_INVARIANT") == "1":
        return DeterminismMode.BATCH_INVARIANT
    elif os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0":
        return DeterminismMode.DETERMINISTIC_SCHEDULING
    else:
        return DeterminismMode.NONE


def get_reproducibility_warnings(mode: DeterminismMode) -> list[str]:
    """
    Get warnings about reproducibility limitations.

    Args:
        mode: Current determinism mode.

    Returns:
        List of warning messages.
    """
    warnings = [
        "vLLM reproducibility requires same hardware and vLLM version.",
        "Different GPU types may produce different results.",
    ]

    if mode == DeterminismMode.NONE:
        warnings.append(
            "No determinism mode is set. Results may vary between runs."
        )

    if mode == DeterminismMode.BATCH_INVARIANT:
        warnings.append(
            "Batch-invariant mode does not guarantee identical results with "
            "different batch sizes, only that batching the same requests differently "
            "produces the same outputs."
        )

    return warnings


def create_runtime_env_for_determinism(
    mode: DeterminismMode,
    base_runtime_env: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a Ray runtime_env dict with determinism settings.

    Args:
        mode: Determinism mode.
        base_runtime_env: Optional base runtime_env to extend.

    Returns:
        runtime_env dict for Ray.
    """
    runtime_env = dict(base_runtime_env) if base_runtime_env else {}

    # Get determinism env vars
    config = DeterminismConfig(mode=mode)
    det_env_vars = config.env_vars

    # Merge with existing env_vars
    existing_env = runtime_env.get("env_vars", {})
    runtime_env["env_vars"] = {**existing_env, **det_env_vars}

    return runtime_env


# Convenience constants
DETERMINISM_ENV_VARS = {
    "batch_invariant": {"VLLM_BATCH_INVARIANT": "1"},
    "deterministic_scheduling": {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
}

# Default recommended mode
DEFAULT_DETERMINISM_MODE = DeterminismMode.BATCH_INVARIANT
