"""Tests for cache functionality."""

import tempfile
from pathlib import Path

import pytest

from labelforge.cache.key import CacheKey, compute_cache_key, compute_output_hash
from labelforge.cache.fs_cache import FilesystemCache


class TestCacheKey:
    """Tests for cache key operations."""

    def test_create_cache_key(self):
        """Cache keys should be creatable."""
        key = compute_cache_key(
            stage_name="caption",
            row_id="lf_1234567890abcdef",
            prompt_hash="abc123",
            model_hash="def456",
            sampling_params_hash="ghi789",
        )
        assert key.stage_name == "caption"
        assert key.row_id == "lf_1234567890abcdef"

    def test_key_string_format(self):
        """Key string should contain all components."""
        key = CacheKey(
            stage_name="caption",
            row_id="lf_1234567890abcdef",
            prompt_hash="abc123",
            model_hash="def456",
            sampling_params_hash="ghi789",
        )
        key_str = key.key_string
        assert "caption" in key_str
        assert "abc123" in key_str
        assert "def456" in key_str

    def test_key_hash_stability(self):
        """Same key should produce same hash."""
        key1 = CacheKey(
            stage_name="caption",
            row_id="lf_1234567890abcdef",
            prompt_hash="abc123",
            model_hash="def456",
            sampling_params_hash="ghi789",
        )
        key2 = CacheKey(
            stage_name="caption",
            row_id="lf_1234567890abcdef",
            prompt_hash="abc123",
            model_hash="def456",
            sampling_params_hash="ghi789",
        )
        assert key1.hash == key2.hash

    def test_key_hash_different(self):
        """Different keys should produce different hashes."""
        key1 = CacheKey(
            stage_name="caption",
            row_id="lf_1234567890abcdef",
            prompt_hash="abc123",
            model_hash="def456",
            sampling_params_hash="ghi789",
        )
        key2 = CacheKey(
            stage_name="score",  # Different stage
            row_id="lf_1234567890abcdef",
            prompt_hash="abc123",
            model_hash="def456",
            sampling_params_hash="ghi789",
        )
        assert key1.hash != key2.hash

    def test_prefix_format(self):
        """Prefix should be 4 hex characters."""
        key = CacheKey(
            stage_name="caption",
            row_id="lf_1234567890abcdef",
            prompt_hash="abc123",
            model_hash="def456",
            sampling_params_hash="ghi789",
        )
        assert len(key.prefix) == 4
        assert all(c in "0123456789abcdef" for c in key.prefix)

    def test_to_dict_roundtrip(self):
        """Key should survive dict roundtrip."""
        original = CacheKey(
            stage_name="caption",
            row_id="lf_1234567890abcdef",
            prompt_hash="abc123",
            model_hash="def456",
            sampling_params_hash="ghi789",
        )
        recovered = CacheKey.from_dict(original.to_dict())
        assert recovered.stage_name == original.stage_name
        assert recovered.row_id == original.row_id
        assert recovered.hash == original.hash


class TestFilesystemCache:
    """Tests for filesystem cache backend."""

    @pytest.fixture
    def cache_dir(self):
        """Create a temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def cache(self, cache_dir):
        """Create a filesystem cache."""
        return FilesystemCache(cache_dir)

    def test_put_and_get(self, cache):
        """Should be able to store and retrieve values."""
        key = CacheKey(
            stage_name="test",
            row_id="lf_1234567890abcdef",
            prompt_hash="p1",
            model_hash="m1",
            sampling_params_hash="s1",
        )
        value = {"result": "test_value", "score": 0.95}
        output_hash = compute_output_hash(value)

        entry = cache.put(key, value, output_hash)
        assert entry.output_hash == output_hash

        retrieved = cache.get(key)
        assert retrieved is not None
        assert retrieved.output_hash == output_hash

    def test_get_value(self, cache):
        """Should be able to retrieve actual value."""
        key = CacheKey(
            stage_name="test",
            row_id="lf_1234567890abcdef",
            prompt_hash="p1",
            model_hash="m1",
            sampling_params_hash="s1",
        )
        value = {"result": "test_value", "score": 0.95}
        output_hash = compute_output_hash(value)

        cache.put(key, value, output_hash)
        entry = cache.get(key)
        
        retrieved_value = cache.get_value(entry)
        assert retrieved_value == value

    def test_exists(self, cache):
        """exists() should correctly report presence."""
        key = CacheKey(
            stage_name="test",
            row_id="lf_1234567890abcdef",
            prompt_hash="p1",
            model_hash="m1",
            sampling_params_hash="s1",
        )
        
        assert not cache.exists(key)
        
        cache.put(key, {"value": 1}, "hash")
        
        assert cache.exists(key)

    def test_delete(self, cache):
        """delete() should remove entry."""
        key = CacheKey(
            stage_name="test",
            row_id="lf_1234567890abcdef",
            prompt_hash="p1",
            model_hash="m1",
            sampling_params_hash="s1",
        )
        
        cache.put(key, {"value": 1}, "hash")
        assert cache.exists(key)
        
        result = cache.delete(key)
        assert result is True
        assert not cache.exists(key)

    def test_get_stats(self, cache):
        """get_stats() should return statistics."""
        key = CacheKey(
            stage_name="test",
            row_id="lf_1234567890abcdef",
            prompt_hash="p1",
            model_hash="m1",
            sampling_params_hash="s1",
        )
        cache.put(key, {"value": 1}, "hash")

        stats = cache.get_stats()
        assert stats["total_entries"] == 1
        assert stats["total_size_bytes"] > 0
