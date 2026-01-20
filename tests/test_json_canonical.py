"""Tests for JSON canonical serialization."""

import pytest
import math

from labelforge.core.json_canonical import (
    canonical_json_dumps,
    canonical_json_loads,
    canonical_json_bytes,
    stable_float_repr,
    to_jsonl_line,
    from_jsonl_lines,
)


class TestCanonicalJsonAdvanced:
    """Advanced tests for canonical JSON."""

    def test_unicode_handling(self):
        """Unicode should be preserved."""
        obj = {"key": "Hello, 世界! 🎉"}
        result = canonical_json_dumps(obj)
        recovered = canonical_json_loads(result)
        assert recovered["key"] == "Hello, 世界! 🎉"

    def test_nested_list_sort(self):
        """Lists should preserve order (not sorted)."""
        obj = {"items": [3, 1, 2]}
        result = canonical_json_dumps(obj)
        recovered = canonical_json_loads(result)
        assert recovered["items"] == [3, 1, 2]  # Order preserved

    def test_deep_nesting(self):
        """Deep nesting should work."""
        obj = {"a": {"b": {"c": {"d": {"e": "deep"}}}}}
        result = canonical_json_dumps(obj)
        recovered = canonical_json_loads(result)
        assert recovered["a"]["b"]["c"]["d"]["e"] == "deep"

    def test_empty_structures(self):
        """Empty structures should serialize correctly."""
        obj = {"empty_dict": {}, "empty_list": [], "empty_str": ""}
        result = canonical_json_dumps(obj)
        recovered = canonical_json_loads(result)
        assert recovered == obj

    def test_boolean_values(self):
        """Boolean values should be lowercase."""
        obj = {"t": True, "f": False}
        result = canonical_json_dumps(obj)
        assert "true" in result
        assert "false" in result

    def test_null_value(self):
        """Null should serialize correctly."""
        obj = {"value": None}
        result = canonical_json_dumps(obj)
        assert "null" in result

    def test_large_numbers(self):
        """Large numbers should be handled."""
        obj = {"big": 9999999999999999}
        result = canonical_json_dumps(obj)
        recovered = canonical_json_loads(result)
        assert recovered["big"] == 9999999999999999

    def test_bytes_matches_string(self):
        """Bytes output should match string."""
        obj = {"key": "value", "number": 42}
        str_result = canonical_json_dumps(obj)
        bytes_result = canonical_json_bytes(obj)
        assert bytes_result.decode("utf-8") == str_result


class TestStableFloatRepr:
    """Tests for stable float representation."""

    def test_typical_floats(self):
        """Typical floats should format correctly."""
        assert stable_float_repr(1.5, precision=6) == "1.500000"
        assert stable_float_repr(0.123456, precision=6) == "0.123456"

    def test_rounding(self):
        """Rounding should work correctly."""
        # 0.123456789 rounded to 6 places
        result = stable_float_repr(0.123456789, precision=6)
        assert result == "0.123457"

    def test_negative_numbers(self):
        """Negative numbers should work."""
        assert stable_float_repr(-1.5, precision=2) == "-1.50"

    def test_zero(self):
        """Zero should work."""
        assert stable_float_repr(0.0, precision=2) == "0.00"

    def test_special_values(self):
        """Special float values should be handled."""
        assert stable_float_repr(float("inf")) == "inf"
        assert stable_float_repr(float("-inf")) == "-inf"
        assert stable_float_repr(float("nan")) == "nan"


class TestJsonlFunctions:
    """Tests for JSONL functions."""

    def test_to_jsonl_line(self):
        """Should convert to JSONL line."""
        obj = {"key": "value", "num": 42}
        line = to_jsonl_line(obj)
        assert line.endswith("\n")
        assert "key" in line
        assert "value" in line

    def test_to_jsonl_line_no_newline(self):
        """Middle of string should not have newline."""
        obj = {"key": "value"}
        line = to_jsonl_line(obj)
        # Should only have one newline at end
        assert line.count("\n") == 1
        assert line.strip()  # Not empty

    def test_from_jsonl_lines(self):
        """Should parse JSONL content."""
        content = '{"a": 1}\n{"a": 2}\n{"a": 3}\n'
        records = from_jsonl_lines(content)
        assert len(records) == 3
        assert records[0]["a"] == 1
        assert records[2]["a"] == 3

    def test_from_jsonl_lines_empty(self):
        """Empty content should return empty list."""
        records = from_jsonl_lines("")
        assert records == []

    def test_from_jsonl_lines_blank_lines(self):
        """Blank lines should be skipped."""
        content = '{"a": 1}\n\n{"a": 2}\n   \n{"a": 3}\n'
        records = from_jsonl_lines(content)
        assert len(records) == 3

    def test_jsonl_roundtrip(self):
        """JSONL should roundtrip correctly."""
        original = [
            {"id": 1, "name": "first"},
            {"id": 2, "name": "second"},
        ]
        lines = "".join(to_jsonl_line(obj) for obj in original)
        recovered = from_jsonl_lines(lines)
        assert recovered == original
