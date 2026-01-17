"""Tests for hashing and canonicalization."""

import pytest

from labelforge.core.json_canonical import (
    canonical_json_dumps,
    canonical_json_loads,
    canonical_json_bytes,
    stable_float_repr,
)
from labelforge.core.row_id import (
    compute_row_id,
    validate_row_id,
    compute_image_hash,
    compute_text_hash,
)
from labelforge.core.schema import DataSource, DataSourceType


class TestCanonicalJson:
    """Tests for canonical JSON serialization."""

    def test_sorted_keys(self):
        """JSON should have sorted keys."""
        obj = {"z": 1, "a": 2, "m": 3}
        result = canonical_json_dumps(obj)
        assert result == '{"a":2,"m":3,"z":1}'

    def test_nested_sorted_keys(self):
        """Nested objects should also have sorted keys."""
        obj = {"b": {"z": 1, "a": 2}, "a": 1}
        result = canonical_json_dumps(obj)
        assert result == '{"a":1,"b":{"a":2,"z":1}}'

    def test_deterministic_output(self):
        """Same input should always produce same output."""
        obj = {"key": "value", "number": 42, "list": [1, 2, 3]}
        results = [canonical_json_dumps(obj) for _ in range(10)]
        assert len(set(results)) == 1

    def test_roundtrip(self):
        """Data should survive roundtrip."""
        original = {"key": "value", "nested": {"a": 1, "b": [1, 2, 3]}}
        json_str = canonical_json_dumps(original)
        recovered = canonical_json_loads(json_str)
        assert recovered == original

    def test_float_representation(self):
        """Floats should be represented consistently."""
        obj = {"value": 1.0}
        result = canonical_json_dumps(obj)
        assert "1.0" in result or "1" in result  # orjson may optimize

    def test_bytes_output_identical(self):
        """Bytes output should match string output."""
        obj = {"key": "value"}
        json_str = canonical_json_dumps(obj)
        json_bytes = canonical_json_bytes(obj)
        assert json_bytes.decode("utf-8") == json_str


class TestStableFloatRepr:
    """Tests for stable float representation."""

    def test_normal_float(self):
        """Normal floats should have fixed precision."""
        result = stable_float_repr(0.1 + 0.2, precision=6)
        assert result == "0.300000"

    def test_infinity(self):
        """Infinity should be handled."""
        assert stable_float_repr(float("inf")) == "inf"
        assert stable_float_repr(float("-inf")) == "-inf"

    def test_nan(self):
        """NaN should be handled."""
        assert stable_float_repr(float("nan")) == "nan"


class TestRowId:
    """Tests for row ID computation."""

    def test_compute_row_id_format(self):
        """Row IDs should have correct format."""
        source = DataSource(
            source_type=DataSourceType.FILE,
            source_uri="file:///path/to/image.jpg",
            source_key="image.jpg",
        )
        row_id = compute_row_id(source)
        assert row_id.startswith("lf_")
        assert len(row_id) == 19  # "lf_" + 16 hex chars

    def test_row_id_stability(self):
        """Same inputs should produce same row ID."""
        source = DataSource(
            source_type=DataSourceType.FILE,
            source_uri="file:///path/to/image.jpg",
            source_key="image.jpg",
        )
        id1 = compute_row_id(source, text="hello")
        id2 = compute_row_id(source, text="hello")
        assert id1 == id2

    def test_row_id_different_inputs(self):
        """Different inputs should produce different row IDs."""
        source = DataSource(
            source_type=DataSourceType.FILE,
            source_uri="file:///path/to/image.jpg",
            source_key="image.jpg",
        )
        id1 = compute_row_id(source, text="hello")
        id2 = compute_row_id(source, text="world")
        assert id1 != id2

    def test_validate_row_id_valid(self):
        """Valid row IDs should pass validation."""
        assert validate_row_id("lf_1234567890abcdef")
        assert validate_row_id("lf_aaaaaaaaaaaaaaaa")

    def test_validate_row_id_invalid(self):
        """Invalid row IDs should fail validation."""
        assert not validate_row_id("invalid")
        assert not validate_row_id("lf_short")
        assert not validate_row_id("lf_1234567890abcdeg")  # 'g' not hex

    def test_override_key(self):
        """Override key should be used directly."""
        source = DataSource(
            source_type=DataSourceType.DATASET,
            source_uri="dataset://test",
        )
        id1 = compute_row_id(source, row_key_override="custom_key_1")
        id2 = compute_row_id(source, row_key_override="custom_key_1")
        id3 = compute_row_id(source, row_key_override="custom_key_2")
        
        assert id1 == id2
        assert id1 != id3


class TestContentHashing:
    """Tests for content hashing."""

    def test_image_hash_stability(self):
        """Same bytes should produce same hash."""
        data = b"test image data"
        hash1 = compute_image_hash(data)
        hash2 = compute_image_hash(data)
        assert hash1 == hash2

    def test_image_hash_different_data(self):
        """Different bytes should produce different hash."""
        hash1 = compute_image_hash(b"data1")
        hash2 = compute_image_hash(b"data2")
        assert hash1 != hash2

    def test_text_hash_stability(self):
        """Same text should produce same hash."""
        text = "hello world"
        hash1 = compute_text_hash(text)
        hash2 = compute_text_hash(text)
        assert hash1 == hash2
