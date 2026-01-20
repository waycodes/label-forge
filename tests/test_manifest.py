"""Tests for manifest system."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from labelforge.core.manifest.run_manifest import (
    RunManifest,
    RunMetadata,
    StageReference,
)
from labelforge.core.manifest.stage_manifest import (
    StageManifest,
    CacheStats,
    TimingStats,
)
from labelforge.core.manifest.row_record import (
    RowRecord,
    RowStatus,
)
from labelforge.core.manifest.hash import (
    compute_manifest_hash,
    compute_file_hash,
    compare_manifests,
)


class TestRunMetadata:
    """Tests for RunMetadata model."""

    def test_create_metadata(self):
        """Metadata should be creatable."""
        meta = RunMetadata(
            run_id="abc12345",
            started_at=datetime.utcnow(),
            run_seed=42,
        )
        assert meta.run_id == "abc12345"
        assert meta.run_seed == 42

    def test_with_git_info(self):
        """Git info should be settable."""
        meta = RunMetadata(
            run_id="abc12345",
            started_at=datetime.utcnow(),
            run_seed=42,
            git_commit="abc123def456",
            git_branch="main",
            git_dirty=False,
        )
        assert meta.git_commit == "abc123def456"
        assert meta.git_dirty is False


class TestStageReference:
    """Tests for StageReference model."""

    def test_create_reference(self):
        """Reference should be creatable."""
        ref = StageReference(
            stage_name="caption",
            stage_type="vlm_caption",
            stage_version="1.0.0",
            stage_hash="abc123",
        )
        assert ref.stage_name == "caption"

    def test_with_dependencies(self):
        """Dependencies should be settable."""
        ref = StageReference(
            stage_name="score",
            stage_type="rubric_score",
            stage_version="1.0.0",
            stage_hash="def456",
            depends_on=["caption"],
        )
        assert "caption" in ref.depends_on


class TestRunManifest:
    """Tests for RunManifest model."""

    @pytest.fixture
    def sample_manifest(self):
        """Create a sample manifest."""
        return RunManifest.create(
            run_id="test123",
            output_dir="/tmp/runs/test123",
            run_seed=42,
            run_name="Test Run",
        )

    def test_create_manifest(self, sample_manifest):
        """Manifest should be creatable."""
        assert sample_manifest.metadata.run_id == "test123"
        assert sample_manifest.metadata.run_seed == 42

    def test_manifest_to_json(self, sample_manifest):
        """Manifest should serialize to JSON."""
        json_str = sample_manifest.to_json()
        assert "test123" in json_str
        assert "42" in json_str

    def test_manifest_save_load(self, sample_manifest):
        """Manifest should roundtrip through file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            sample_manifest.save(path)
            
            loaded = RunManifest.load(path)
            assert loaded.metadata.run_id == sample_manifest.metadata.run_id
            assert loaded.metadata.run_seed == sample_manifest.metadata.run_seed


class TestCacheStats:
    """Tests for CacheStats model."""

    def test_create_stats(self):
        """Stats should be creatable."""
        stats = CacheStats(
            hits=100,
            misses=20,
            bytes_read=10000,
            bytes_written=2000,
        )
        assert stats.hits == 100
        assert stats.misses == 20

    def test_compute_hit_rate(self):
        """Hit rate should be computed correctly."""
        stats = CacheStats.compute(
            hits=80,
            misses=20,
            bytes_read=8000,
            bytes_written=2000,
        )
        assert stats.hit_rate == 0.8

    def test_zero_hit_rate(self):
        """Zero requests should have 0 hit rate."""
        stats = CacheStats.compute(
            hits=0,
            misses=0,
            bytes_read=0,
            bytes_written=0,
        )
        assert stats.hit_rate == 0.0


class TestStageManifest:
    """Tests for StageManifest model."""

    @pytest.fixture
    def sample_stage_manifest(self):
        """Create a sample stage manifest."""
        return StageManifest.create(
            stage_name="caption",
            stage_type="vlm_caption",
            stage_version="1.0.0",
            run_id="test123",
            stage_index=0,
            stage_config_hash="abc123",
            output_dataset_path="/tmp/output",
        )

    def test_create_stage_manifest(self, sample_stage_manifest):
        """Stage manifest should be creatable."""
        assert sample_stage_manifest.stage_name == "caption"
        assert sample_stage_manifest.status == "running"

    def test_stage_manifest_to_json(self, sample_stage_manifest):
        """Stage manifest should serialize to JSON."""
        json_str = sample_stage_manifest.to_json()
        assert "caption" in json_str

    def test_stage_manifest_save_load(self, sample_stage_manifest):
        """Stage manifest should roundtrip through file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "stage_manifest.json"
            sample_stage_manifest.save(path)
            
            loaded = StageManifest.load(path)
            assert loaded.stage_name == sample_stage_manifest.stage_name


class TestRowRecord:
    """Tests for RowRecord model."""

    def test_create_success_record(self):
        """Success record should be creatable."""
        record = RowRecord.success(
            row_id="lf_1234567890abcdef",
            stage_name="caption",
            input_hash="abc123",
            prompt_hash="def456",
            model_hash="ghi789",
            sampling_params_hash="jkl012",
            output_hash="mno345",
            output_ref="/cache/output.json",
            latency_ms=150.5,
        )
        assert record.status == RowStatus.SUCCESS
        assert record.latency_ms == 150.5

    def test_create_error_record(self):
        """Error record should be creatable."""
        record = RowRecord.error(
            row_id="lf_1234567890abcdef",
            stage_name="caption",
            input_hash="abc123",
            prompt_hash="def456",
            model_hash="ghi789",
            sampling_params_hash="jkl012",
            error_type="ValueError",
            error_message="Invalid input",
        )
        assert record.status == RowStatus.ERROR
        assert record.error_type == "ValueError"

    def test_create_cached_record(self):
        """Cached record should be creatable."""
        record = RowRecord.cached(
            row_id="lf_1234567890abcdef",
            stage_name="caption",
            input_hash="abc123",
            prompt_hash="def456",
            model_hash="ghi789",
            sampling_params_hash="jkl012",
            output_hash="mno345",
            output_ref="/cache/cached.json",
        )
        assert record.status == RowStatus.CACHED
        assert record.latency_ms == 0.0

    def test_create_skipped_record(self):
        """Skipped record should be creatable."""
        record = RowRecord.skipped(
            row_id="lf_1234567890abcdef",
            stage_name="caption",
            input_hash="abc123",
            prompt_hash="def456",
            model_hash="ghi789",
            sampling_params_hash="jkl012",
            reason="Missing image",
        )
        assert record.status == RowStatus.SKIPPED
        assert record.error_message == "Missing image"

    def test_to_dict(self):
        """Record should convert to dict."""
        record = RowRecord.success(
            row_id="lf_1234567890abcdef",
            stage_name="caption",
            input_hash="abc123",
            prompt_hash="def456",
            model_hash="ghi789",
            sampling_params_hash="jkl012",
            output_hash="mno345",
            output_ref="/cache/output.json",
            latency_ms=100.0,
        )
        data = record.to_dict()
        assert data["row_id"] == "lf_1234567890abcdef"
        assert data["status"] == "success"


class TestManifestHashing:
    """Tests for manifest hashing."""

    def test_manifest_hash_stability(self):
        """Same manifest should produce same hash."""
        manifest = RunManifest.create(
            run_id="test123",
            output_dir="/tmp/runs/test123",
            run_seed=42,
        )
        hash1 = compute_manifest_hash(manifest)
        hash2 = compute_manifest_hash(manifest)
        assert hash1 == hash2

    def test_manifest_hash_different_seeds(self):
        """Different seeds should produce different hashes."""
        m1 = RunManifest.create(
            run_id="test123",
            output_dir="/tmp/runs/test123",
            run_seed=42,
        )
        m2 = RunManifest.create(
            run_id="test123",
            output_dir="/tmp/runs/test123",
            run_seed=123,
        )
        assert compute_manifest_hash(m1) != compute_manifest_hash(m2)

    def test_file_hash(self):
        """File hash should work."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("test content")
            f.flush()
            
            hash1 = compute_file_hash(Path(f.name))
            hash2 = compute_file_hash(Path(f.name))
            assert hash1 == hash2

    def test_compare_manifests(self):
        """Manifest comparison should work."""
        m1 = RunManifest.create(
            run_id="test123",
            output_dir="/tmp/runs/test123",
            run_seed=42,
            git_commit="abc123",
        )
        m2 = RunManifest.create(
            run_id="test456",  # Different ID but same config
            output_dir="/tmp/runs/test456",
            run_seed=42,
            git_commit="abc123",
        )
        
        result = compare_manifests(m1, m2)
        assert result["run_seed_match"] is True
        assert result["git_commit_match"] is True
