"""
Stable row_id computation for cache keys and manifest references.

Row IDs are content-derived hashes that uniquely identify a data row
across runs, enabling caching and reproducibility.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import xxhash

if TYPE_CHECKING:
    from labelforge.core.schema import DataSource

# Row ID format: lf_{hash} where hash is 16 hex characters
ROW_ID_PREFIX = "lf_"
ROW_ID_PATTERN = re.compile(r"^lf_[0-9a-f]{16}$")


def compute_row_id(
    source: DataSource,
    text: str | None = None,
    image_hash: str | None = None,
    row_key_override: str | None = None,
) -> str:
    """
    Compute a stable row ID from source identifiers and content.

    The row ID is a content-addressed hash that remains stable across runs
    as long as the source data doesn't change.

    Args:
        source: Data source provenance information.
        text: Optional text content to include in hash.
        image_hash: Optional pre-computed image content hash.
        row_key_override: Optional explicit key (bypasses content hashing).

    Returns:
        A stable row ID in format "lf_{16-char-hex}".

    Examples:
        >>> from labelforge.core.schema import DataSource, DataSourceType
        >>> source = DataSource(
        ...     source_type=DataSourceType.FILE,
        ...     source_uri="file:///data/images/cat.jpg",
        ...     source_key="cat.jpg",
        ... )
        >>> row_id = compute_row_id(source, image_hash="abc123")
        >>> row_id.startswith("lf_")
        True
    """
    if row_key_override:
        # Use override directly (for external datasets with stable keys)
        hash_input = row_key_override
    else:
        # Build deterministic hash input from components
        components = [
            source.source_type.value,
            source.source_uri,
            source.source_key or "",
            source.source_version or "",
        ]

        # Include content hashes if available
        if text is not None:
            components.append(f"text:{text}")
        if image_hash is not None:
            components.append(f"image:{image_hash}")

        hash_input = "|".join(components)

    # Use xxhash for fast, stable hashing
    hash_bytes = xxhash.xxh64(hash_input.encode("utf-8")).digest()
    hash_hex = hash_bytes.hex()

    return f"{ROW_ID_PREFIX}{hash_hex}"


def validate_row_id(row_id: str) -> bool:
    """
    Validate that a string is a valid LabelForge row ID.

    Args:
        row_id: The string to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> validate_row_id("lf_1234567890abcdef")
        True
        >>> validate_row_id("invalid")
        False
    """
    return bool(ROW_ID_PATTERN.match(row_id))


def compute_image_hash(image_bytes: bytes) -> str:
    """
    Compute a content hash for image bytes.

    Args:
        image_bytes: Raw image bytes.

    Returns:
        Hex-encoded xxhash digest.
    """
    return xxhash.xxh64(image_bytes).hexdigest()


def compute_text_hash(text: str) -> str:
    """
    Compute a content hash for text.

    Args:
        text: Text string.

    Returns:
        Hex-encoded xxhash digest.
    """
    return xxhash.xxh64(text.encode("utf-8")).hexdigest()


def derive_child_row_id(parent_row_id: str, suffix: str) -> str:
    """
    Derive a child row ID from a parent (for synthetic data generation).

    Args:
        parent_row_id: The parent row's ID.
        suffix: A distinguishing suffix (e.g., "synth_0", "neg_1").

    Returns:
        A new row ID derived from the parent.

    Examples:
        >>> parent = "lf_1234567890abcdef"
        >>> child = derive_child_row_id(parent, "synth_0")
        >>> child.startswith("lf_")
        True
        >>> child != parent
        True
    """
    hash_input = f"{parent_row_id}:{suffix}"
    hash_bytes = xxhash.xxh64(hash_input.encode("utf-8")).digest()
    return f"{ROW_ID_PREFIX}{hash_bytes.hex()}"
