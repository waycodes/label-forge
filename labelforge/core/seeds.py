"""
Global seed policy for deterministic execution.

Provides hierarchical seed derivation: run → stage → row.
All RNG usage should derive seeds from this policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import xxhash


@dataclass(frozen=True)
class SeedPolicy:
    """
    Seed policy for a pipeline run.

    Seeds are derived hierarchically to ensure:
    - Same run_seed produces same results across reruns
    - Each stage gets a unique but reproducible seed
    - Each row gets a unique but reproducible seed
    """

    run_seed: int
    """Base seed for the entire run. Must be explicitly set."""

    def __post_init__(self) -> None:
        """Validate seed value."""
        if not (0 <= self.run_seed < 2**32):
            raise ValueError(f"run_seed must be in [0, 2^32), got {self.run_seed}")

    def derive_stage_seed(self, stage_name: str) -> int:
        """
        Derive a stage-specific seed.

        Args:
            stage_name: Unique name of the stage.

        Returns:
            Deterministic seed for this stage.

        Examples:
            >>> policy = SeedPolicy(run_seed=42)
            >>> policy.derive_stage_seed("caption")
            2841326441
            >>> policy.derive_stage_seed("caption")  # Same input, same output
            2841326441
        """
        return derive_stage_seed(self.run_seed, stage_name)

    def derive_row_seed(self, stage_name: str, row_id: str) -> int:
        """
        Derive a row-specific seed.

        Args:
            stage_name: Name of the current stage.
            row_id: Unique row identifier.

        Returns:
            Deterministic seed for this row in this stage.

        Examples:
            >>> policy = SeedPolicy(run_seed=42)
            >>> policy.derive_row_seed("caption", "lf_1234567890abcdef")
            3856789012
        """
        stage_seed = self.derive_stage_seed(stage_name)
        return derive_row_seed(stage_seed, row_id)


def derive_stage_seed(run_seed: int, stage_name: str) -> int:
    """
    Derive a stage-specific seed from the run seed.

    Args:
        run_seed: Base run seed.
        stage_name: Unique stage name.

    Returns:
        32-bit unsigned integer seed.
    """
    hash_input = f"stage:{run_seed}:{stage_name}"
    hash_value = xxhash.xxh32(hash_input.encode("utf-8")).intdigest()
    return hash_value


def derive_row_seed(stage_seed: int, row_id: str) -> int:
    """
    Derive a row-specific seed from the stage seed.

    Args:
        stage_seed: Stage seed (from derive_stage_seed).
        row_id: Unique row identifier.

    Returns:
        32-bit unsigned integer seed.
    """
    hash_input = f"row:{stage_seed}:{row_id}"
    hash_value = xxhash.xxh32(hash_input.encode("utf-8")).intdigest()
    return hash_value


def derive_sampling_seed(row_seed: int, attempt: int = 0) -> int:
    """
    Derive a sampling seed for model inference.

    Args:
        row_seed: Row seed (from derive_row_seed).
        attempt: Retry attempt number (for fault tolerance).

    Returns:
        32-bit unsigned integer seed for model sampling.
    """
    hash_input = f"sample:{row_seed}:{attempt}"
    hash_value = xxhash.xxh32(hash_input.encode("utf-8")).intdigest()
    return hash_value


def set_all_seeds(seed: int) -> None:
    """
    Set seeds for all common RNG sources.

    This sets seeds for:
    - Python's random module
    - NumPy (if available)
    - PyTorch (if available)

    Args:
        seed: Seed value to set.
    """
    import random

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def validate_seed(seed: int | None) -> int:
    """
    Validate and normalize a seed value.

    Args:
        seed: Seed to validate (None generates a random seed).

    Returns:
        Validated seed value.

    Raises:
        ValueError: If seed is out of valid range.
    """
    if seed is None:
        import random

        return random.randint(0, 2**32 - 1)

    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got {type(seed).__name__}")

    if not (0 <= seed < 2**32):
        raise ValueError(f"Seed must be in [0, 2^32), got {seed}")

    return seed
