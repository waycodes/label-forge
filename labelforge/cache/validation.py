"""
Cache validation and invalidation.

Ensures cached results are valid and invalidates stale entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from labelforge.cache.key import CacheKey, compute_output_hash
from labelforge.cache.store import CacheEntry, CacheStore


@dataclass
class ValidationResult:
    """Result of cache validation."""

    valid: bool
    reason: str | None = None
    expected_hash: str | None = None
    actual_hash: str | None = None


def validate_cache_entry(
    entry: CacheEntry,
    store: CacheStore,
    verify_content: bool = True,
) -> ValidationResult:
    """
    Validate a cache entry.

    Checks:
    - Entry exists
    - Value can be retrieved
    - Output hash matches (if verify_content)

    Args:
        entry: Cache entry to validate.
        store: Cache store to retrieve value from.
        verify_content: Whether to verify content hash.

    Returns:
        ValidationResult indicating validity.
    """
    try:
        value = store.get_value(entry)
    except Exception as e:
        return ValidationResult(valid=False, reason=f"Failed to retrieve value: {e}")

    if verify_content:
        actual_hash = compute_output_hash(value)
        if actual_hash != entry.output_hash:
            return ValidationResult(
                valid=False,
                reason="Output hash mismatch",
                expected_hash=entry.output_hash,
                actual_hash=actual_hash,
            )

    return ValidationResult(valid=True)


def check_cache_validity(
    key: CacheKey,
    current_prompt_hash: str,
    current_model_hash: str,
    current_sampling_hash: str,
) -> bool:
    """
    Check if a cache key would be valid for current config.

    This checks if the fingerprints in the key match the current config.

    Args:
        key: Cache key to check.
        current_prompt_hash: Current prompt fingerprint.
        current_model_hash: Current model fingerprint.
        current_sampling_hash: Current sampling params fingerprint.

    Returns:
        True if cache key matches current config.
    """
    return (
        key.prompt_hash == current_prompt_hash
        and key.model_hash == current_model_hash
        and key.sampling_params_hash == current_sampling_hash
    )


def find_invalidated_entries(
    store: CacheStore,
    stage_name: str,
    current_prompt_hash: str,
    current_model_hash: str,
) -> list[CacheKey]:
    """
    Find cache entries that should be invalidated.

    An entry is invalid if its prompt or model hash doesn't
    match the current config.

    Args:
        store: Cache store.
        stage_name: Stage name to check.
        current_prompt_hash: Current prompt hash.
        current_model_hash: Current model hash.

    Returns:
        List of cache keys that should be invalidated.
    """
    # This is a placeholder - actual implementation would query
    # the store's metadata database
    return []


def invalidate_by_lineage(
    store: CacheStore,
    source_stage: str,
    source_row_ids: set[str],
) -> int:
    """
    Invalidate cache entries whose input came from invalidated source rows.

    This implements lineage-based invalidation for downstream stages.

    Args:
        store: Cache store.
        source_stage: Name of the source stage.
        source_row_ids: Set of source row IDs that were invalidated.

    Returns:
        Number of entries invalidated.
    """
    # Placeholder for lineage-based invalidation
    # Would need to track input-output relationships
    return 0


class CacheValidator:
    """
    Cache validator for ensuring cache integrity.

    Tracks expected fingerprints and validates entries against them.
    """

    def __init__(
        self,
        store: CacheStore,
        prompt_hash: str,
        model_hash: str,
        sampling_params_hash: str,
    ):
        """
        Initialize validator.

        Args:
            store: Cache store.
            prompt_hash: Expected prompt hash.
            model_hash: Expected model hash.
            sampling_params_hash: Expected sampling params hash.
        """
        self.store = store
        self.prompt_hash = prompt_hash
        self.model_hash = model_hash
        self.sampling_params_hash = sampling_params_hash

    def is_valid_key(self, key: CacheKey) -> bool:
        """Check if a cache key matches expected fingerprints."""
        return check_cache_validity(
            key,
            self.prompt_hash,
            self.model_hash,
            self.sampling_params_hash,
        )

    def get_if_valid(self, key: CacheKey) -> CacheEntry | None:
        """
        Get cache entry if it exists and is valid.

        Args:
            key: Cache key.

        Returns:
            CacheEntry if valid, None otherwise.
        """
        if not self.is_valid_key(key):
            return None

        entry = self.store.get(key)
        if entry is None:
            return None

        validation = validate_cache_entry(entry, self.store, verify_content=False)
        if not validation.valid:
            return None

        return entry

    def get_value_if_valid(self, key: CacheKey) -> Any | None:
        """
        Get cached value if entry exists and is valid.

        Args:
            key: Cache key.

        Returns:
            Cached value if valid, None otherwise.
        """
        entry = self.get_if_valid(key)
        if entry is None:
            return None

        return self.store.get_value(entry)
