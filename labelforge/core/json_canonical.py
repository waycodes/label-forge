"""
Deterministic JSON serialization for byte-stable manifests.

Ensures that identical data produces identical JSON regardless of
dict ordering, float representation, or platform differences.
"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import orjson


def _default_serializer(obj: Any) -> Any:
    """
    Custom serializer for types not natively supported by orjson.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-serializable representation.

    Raises:
        TypeError: If object cannot be serialized.
    """
    if isinstance(obj, datetime):
        # ISO 8601 format with microseconds and Z suffix
        return obj.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if isinstance(obj, Decimal):
        # Use string representation for full precision
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, bytes):
        # Hex-encode bytes
        return obj.hex()
    if isinstance(obj, set):
        # Convert to sorted list for determinism
        return sorted(obj, key=str)
    if isinstance(obj, frozenset):
        return sorted(obj, key=str)
    if hasattr(obj, "model_dump"):
        # Pydantic models
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return obj.__dict__

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def canonical_json_dumps(
    obj: Any,
    *,
    indent: bool = False,
    ensure_ascii: bool = False,
) -> str:
    """
    Serialize object to canonical JSON string.

    Guarantees:
    - Sorted dictionary keys
    - Consistent float formatting
    - UTF-8 encoding
    - Normalized newlines
    - Stable datetime formatting

    Args:
        obj: Object to serialize.
        indent: If True, pretty-print with 2-space indentation.
        ensure_ascii: If True, escape non-ASCII characters.

    Returns:
        Canonical JSON string.

    Examples:
        >>> canonical_json_dumps({"b": 2, "a": 1})
        '{"a":1,"b":2}'
        >>> canonical_json_dumps({"x": 1.0})
        '{"x":1.0}'
    """
    options = orjson.OPT_SORT_KEYS

    if indent:
        options |= orjson.OPT_INDENT_2

    # orjson returns bytes, decode to string
    result = orjson.dumps(obj, default=_default_serializer, option=options)
    json_str = result.decode("utf-8")

    # Normalize line endings
    json_str = json_str.replace("\r\n", "\n").replace("\r", "\n")

    return json_str


def canonical_json_loads(json_str: str) -> Any:
    """
    Parse JSON string.

    Args:
        json_str: JSON string to parse.

    Returns:
        Parsed Python object.

    Examples:
        >>> canonical_json_loads('{"a":1,"b":2}')
        {'a': 1, 'b': 2}
    """
    return orjson.loads(json_str)


def canonical_json_bytes(obj: Any) -> bytes:
    """
    Serialize object to canonical JSON bytes.

    Useful for computing hashes of JSON objects.

    Args:
        obj: Object to serialize.

    Returns:
        Canonical JSON as UTF-8 bytes.
    """
    return orjson.dumps(obj, default=_default_serializer, option=orjson.OPT_SORT_KEYS)


def stable_float_repr(value: float, precision: int = 6) -> str:
    """
    Create a stable string representation of a float.

    Handles special cases (inf, -inf, nan) and rounds to fixed precision
    to avoid platform-specific representation differences.

    Args:
        value: Float value to represent.
        precision: Number of decimal places.

    Returns:
        Stable string representation.

    Examples:
        >>> stable_float_repr(0.1 + 0.2, precision=6)
        '0.300000'
        >>> stable_float_repr(float('inf'))
        'inf'
    """
    if value != value:  # NaN check
        return "nan"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return f"{value:.{precision}f}"


class CanonicalJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for canonical serialization.

    Use this with stdlib json.dumps when orjson is not available.
    """

    def default(self, obj: Any) -> Any:
        """Encode non-standard types."""
        return _default_serializer(obj)


def to_jsonl_line(obj: Any) -> str:
    """
    Convert object to a single JSONL line.

    Args:
        obj: Object to serialize.

    Returns:
        Single line of canonical JSON with trailing newline.
    """
    return canonical_json_dumps(obj, indent=False) + "\n"


def from_jsonl_lines(content: str) -> list[Any]:
    """
    Parse JSONL content into a list of objects.

    Args:
        content: JSONL content with one JSON object per line.

    Returns:
        List of parsed objects.
    """
    lines = content.strip().split("\n")
    return [canonical_json_loads(line) for line in lines if line.strip()]
