"""
Cache store interface.

Abstract interface for cache backends with metadata and blob storage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from labelforge.cache.key import CacheKey


@dataclass
class CacheEntry:
    """A cached entry with metadata."""

    key: CacheKey
    value_ref: str  # Reference to stored value (path or URI)
    output_hash: str  # Hash of the cached output
    created_at: datetime = field(default_factory=datetime.utcnow)
    accessed_at: datetime | None = None
    access_count: int = 0
    size_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key.to_dict(),
            "value_ref": self.value_ref,
            "output_hash": self.output_hash,
            "created_at": self.created_at.isoformat(),
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else None,
            "access_count": self.access_count,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }


class CacheStore(ABC):
    """
    Abstract cache store interface.

    Implementations handle both metadata and blob storage.
    """

    @abstractmethod
    def get(self, key: CacheKey) -> CacheEntry | None:
        """
        Get a cached entry by key.

        Args:
            key: Cache key.

        Returns:
            CacheEntry if found, None otherwise.
        """
        ...

    @abstractmethod
    def put(
        self,
        key: CacheKey,
        value: Any,
        output_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        """
        Store a value in the cache.

        Args:
            key: Cache key.
            value: Value to store.
            output_hash: Hash of the value for verification.
            metadata: Optional additional metadata.

        Returns:
            Created CacheEntry.
        """
        ...

    @abstractmethod
    def exists(self, key: CacheKey) -> bool:
        """
        Check if a key exists in the cache.

        Args:
            key: Cache key.

        Returns:
            True if exists, False otherwise.
        """
        ...

    @abstractmethod
    def delete(self, key: CacheKey) -> bool:
        """
        Delete an entry from the cache.

        Args:
            key: Cache key.

        Returns:
            True if deleted, False if not found.
        """
        ...

    def bulk_get(self, keys: list[CacheKey]) -> dict[str, CacheEntry | None]:
        """
        Get multiple entries by key.

        Default implementation calls get() for each key.
        Subclasses may override for efficiency.

        Args:
            keys: List of cache keys.

        Returns:
            Dict mapping key hash to entry or None.
        """
        return {key.hash: self.get(key) for key in keys}

    def bulk_exists(self, keys: list[CacheKey]) -> dict[str, bool]:
        """
        Check if multiple keys exist.

        Default implementation calls exists() for each key.
        Subclasses may override for efficiency.

        Args:
            keys: List of cache keys.

        Returns:
            Dict mapping key hash to existence boolean.
        """
        return {key.hash: self.exists(key) for key in keys}

    @abstractmethod
    def get_value(self, entry: CacheEntry) -> Any:
        """
        Retrieve the actual value from a cache entry.

        Args:
            entry: Cache entry with value reference.

        Returns:
            The cached value.
        """
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with stats like total entries, size, etc.
        """
        ...


class CacheStoreProtocol(Protocol):
    """Protocol for cache store implementations."""

    def get(self, key: CacheKey) -> CacheEntry | None:
        ...

    def put(
        self,
        key: CacheKey,
        value: Any,
        output_hash: str,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        ...

    def exists(self, key: CacheKey) -> bool:
        ...

    def delete(self, key: CacheKey) -> bool:
        ...

    def get_value(self, entry: CacheEntry) -> Any:
        ...
