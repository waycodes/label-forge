"""
Cache key format for content-addressed caching.

Keys are composable hashes that uniquely identify cached results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import xxhash

from labelforge.core.json_canonical import canonical_json_bytes


@dataclass(frozen=True)
class CacheKey:
    """
    Content-addressed cache key.

    Keys include all factors that affect the cached result.
    """

    stage_name: str
    row_id: str
    prompt_hash: str
    model_hash: str
    sampling_params_hash: str
    code_hash: str | None = None

    @property
    def key_string(self) -> str:
        """Get the full cache key string."""
        components = [
            self.stage_name,
            self.row_id,
            self.prompt_hash,
            self.model_hash,
            self.sampling_params_hash,
        ]
        if self.code_hash:
            components.append(self.code_hash)

        return ":".join(components)

    @property
    def hash(self) -> str:
        """Get a short hash for storage paths."""
        return xxhash.xxh64(self.key_string.encode("utf-8")).hexdigest()

    @property
    def prefix(self) -> str:
        """Get prefix for sharded storage (first 4 hex chars)."""
        return self.hash[:4]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "stage_name": self.stage_name,
            "row_id": self.row_id,
            "prompt_hash": self.prompt_hash,
            "model_hash": self.model_hash,
            "sampling_params_hash": self.sampling_params_hash,
            "code_hash": self.code_hash,
            "key_string": self.key_string,
            "hash": self.hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheKey:
        """Create from dictionary."""
        return cls(
            stage_name=data["stage_name"],
            row_id=data["row_id"],
            prompt_hash=data["prompt_hash"],
            model_hash=data["model_hash"],
            sampling_params_hash=data["sampling_params_hash"],
            code_hash=data.get("code_hash"),
        )


def compute_cache_key(
    stage_name: str,
    row_id: str,
    prompt_hash: str,
    model_hash: str,
    sampling_params_hash: str,
    code_hash: str | None = None,
) -> CacheKey:
    """
    Compute a cache key from components.

    Args:
        stage_name: Name of the stage.
        row_id: Row identifier.
        prompt_hash: Hash of prompt.
        model_hash: Hash of model config.
        sampling_params_hash: Hash of sampling parameters.
        code_hash: Optional hash of stage code.

    Returns:
        CacheKey instance.

    Example:
        >>> key = compute_cache_key(
        ...     stage_name="caption",
        ...     row_id="lf_1234567890abcdef",
        ...     prompt_hash="abc123",
        ...     model_hash="def456",
        ...     sampling_params_hash="ghi789",
        ... )
        >>> key.stage_name
        'caption'
    """
    return CacheKey(
        stage_name=stage_name,
        row_id=row_id,
        prompt_hash=prompt_hash,
        model_hash=model_hash,
        sampling_params_hash=sampling_params_hash,
        code_hash=code_hash,
    )


def compute_row_input_hash(row: dict[str, Any], fields: list[str] | None = None) -> str:
    """
    Compute hash of row input for cache key.

    Args:
        row: Input row dict.
        fields: Optional list of fields to include. If None, uses all fields.

    Returns:
        Hex-encoded hash.
    """
    if fields:
        data = {k: v for k, v in row.items() if k in fields}
    else:
        # Exclude transient fields
        excluded = {"sampling_params", "_metadata"}
        data = {k: v for k, v in row.items() if k not in excluded}

    json_bytes = canonical_json_bytes(data)
    return xxhash.xxh64(json_bytes).hexdigest()


def compute_output_hash(output: dict[str, Any]) -> str:
    """
    Compute hash of output for verification.

    Args:
        output: Output data dict.

    Returns:
        Hex-encoded hash.
    """
    json_bytes = canonical_json_bytes(output)
    return xxhash.xxh64(json_bytes).hexdigest()
