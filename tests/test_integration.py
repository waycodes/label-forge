"""
Integration tests for I/O utilities and replay planner.

Tests dataset read/write and manifest replay planning.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from labelforge.core.manifest.replay_planner import (
    ManifestReader,
    ReplayMode,
    ReplayPlanner,
    create_replay_plan,
)
from labelforge.core.manifest.run_manifest import RunManifest, StageReference
from labelforge.io.dataset_rw import (
    get_schema_fingerprint,
    read_jsonl_manifest,
    validate_schema_compatibility,
    write_jsonl_manifest,
)


class TestDatasetIO:
    """Tests for dataset I/O utilities."""

    def test_jsonl_write_and_read(self, tmp_path: Path) -> None:
        """Test writing and reading JSONL manifest."""
        records = [
            {"row_id": "1", "status": "ok"},
            {"row_id": "2", "status": "error"},
            {"row_id": "3", "status": "ok"},
        ]

        manifest_path = tmp_path / "manifest.jsonl"
        write_jsonl_manifest(records, manifest_path)

        # Read back
        loaded = read_jsonl_manifest(manifest_path)

        assert len(loaded) == 3
        assert loaded[0]["row_id"] == "1"
        assert loaded[1]["status"] == "error"
        assert loaded[2]["row_id"] == "3"

    def test_jsonl_append_mode(self, tmp_path: Path) -> None:
        """Test appending to JSONL manifest."""
        manifest_path = tmp_path / "manifest.jsonl"

        # Write initial records
        write_jsonl_manifest([{"row_id": "1"}], manifest_path)

        # Append more
        write_jsonl_manifest([{"row_id": "2"}], manifest_path, append=True)

        # Read back
        loaded = read_jsonl_manifest(manifest_path)
        assert len(loaded) == 2

    def test_jsonl_empty_file(self, tmp_path: Path) -> None:
        """Test reading empty JSONL file."""
        manifest_path = tmp_path / "empty.jsonl"
        manifest_path.write_text("")

        loaded = read_jsonl_manifest(manifest_path)
        assert loaded == []

    def test_jsonl_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Test that write_jsonl_manifest creates parent directories."""
        nested_path = tmp_path / "a" / "b" / "c" / "manifest.jsonl"
        write_jsonl_manifest([{"test": True}], nested_path)

        assert nested_path.exists()


class TestSchemaValidation:
    """Tests for schema validation."""

    @pytest.fixture(autouse=True)
    def skip_without_pyarrow(self) -> None:
        """Skip tests if pyarrow is not installed."""
        pytest.importorskip("pyarrow")

    def test_validate_schema_compatibility_match(self) -> None:
        """Test validation with matching schemas."""
        import pyarrow as pa

        actual = pa.schema([("name", pa.string()), ("age", pa.int64())])
        expected = pa.schema([("name", pa.string())])

        errors = validate_schema_compatibility(actual, expected)
        assert errors == []

    def test_validate_schema_compatibility_missing_field(self) -> None:
        """Test validation with missing required field."""
        import pyarrow as pa

        actual = pa.schema([("name", pa.string())])
        expected = pa.schema([("name", pa.string()), ("age", pa.int64())])

        errors = validate_schema_compatibility(actual, expected)
        assert len(errors) == 1
        assert "age" in errors[0]

    def test_validate_schema_compatibility_strict_type(self) -> None:
        """Test strict validation with type mismatch."""
        import pyarrow as pa

        actual = pa.schema([("age", pa.string())])  # string instead of int
        expected = pa.schema([("age", pa.int64())])

        errors = validate_schema_compatibility(actual, expected, strict=True)
        assert len(errors) == 1
        assert "type mismatch" in errors[0]

    def test_validate_schema_compatibility_strict_extra(self) -> None:
        """Test strict validation with extra fields."""
        import pyarrow as pa

        actual = pa.schema([("name", pa.string()), ("extra", pa.string())])
        expected = pa.schema([("name", pa.string())])

        errors = validate_schema_compatibility(actual, expected, strict=True)
        assert len(errors) == 1
        assert "extra" in errors[0]

    def test_schema_fingerprint_stable(self) -> None:
        """Test that schema fingerprint is stable."""
        import pyarrow as pa

        schema1 = pa.schema([("name", pa.string()), ("age", pa.int64())])
        schema2 = pa.schema([("name", pa.string()), ("age", pa.int64())])

        fp1 = get_schema_fingerprint(schema1)
        fp2 = get_schema_fingerprint(schema2)

        assert fp1 == fp2
        assert len(fp1) == 16  # xxhash64 is 16 hex chars

    def test_schema_fingerprint_differs_for_different_schema(self) -> None:
        """Test that different schemas have different fingerprints."""
        import pyarrow as pa

        schema1 = pa.schema([("name", pa.string())])
        schema2 = pa.schema([("name", pa.string()), ("age", pa.int64())])

        assert get_schema_fingerprint(schema1) != get_schema_fingerprint(schema2)


class TestManifestReader:
    """Tests for ManifestReader."""

    @pytest.fixture
    def sample_manifest(self, tmp_path: Path) -> tuple[Path, RunManifest]:
        """Create a sample manifest for testing."""
        manifest = RunManifest.create(
            run_id="run_test_123",
            output_dir=str(tmp_path / "output"),
            run_seed=42,
            run_name="Test Run",
        )

        # Add stages
        manifest = manifest.model_copy(
            update={
                "stages": [
                    StageReference(
                        stage_name="caption",
                        stage_type="vlm_caption",
                        stage_version="1.0.0",
                        stage_hash="abc123",
                        depends_on=[],
                    ),
                    StageReference(
                        stage_name="score",
                        stage_type="rubric_score",
                        stage_version="1.0.0",
                        stage_hash="def456",
                        depends_on=["caption"],
                    ),
                ],
                "stage_manifests": {
                    "caption": str(tmp_path / "output" / "caption" / "manifest.jsonl"),
                },
            }
        )

        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        return manifest_path, manifest

    def test_manifest_reader_load(
        self, sample_manifest: tuple[Path, RunManifest]
    ) -> None:
        """Test loading manifest with reader."""
        manifest_path, _ = sample_manifest
        reader = ManifestReader(manifest_path)

        assert reader.run_id == "run_test_123"
        assert reader.run_seed == 42

    def test_manifest_reader_stages(
        self, sample_manifest: tuple[Path, RunManifest]
    ) -> None:
        """Test accessing stages via reader."""
        manifest_path, _ = sample_manifest
        reader = ManifestReader(manifest_path)

        assert len(reader.stages) == 2
        assert reader.stages[0].stage_name == "caption"
        assert reader.stages[1].stage_name == "score"

    def test_manifest_reader_get_stage(
        self, sample_manifest: tuple[Path, RunManifest]
    ) -> None:
        """Test getting stage by name."""
        manifest_path, _ = sample_manifest
        reader = ManifestReader(manifest_path)

        stage = reader.get_stage("score")
        assert stage is not None
        assert stage.stage_type == "rubric_score"
        assert stage.depends_on == ["caption"]

        assert reader.get_stage("nonexistent") is None

    def test_manifest_reader_validation(
        self, sample_manifest: tuple[Path, RunManifest]
    ) -> None:
        """Test manifest validation."""
        manifest_path, _ = sample_manifest
        reader = ManifestReader(manifest_path)

        errors = reader.validate_manifest()
        assert errors == []

    def test_manifest_reader_output_path(
        self, sample_manifest: tuple[Path, RunManifest]
    ) -> None:
        """Test getting stage output paths."""
        manifest_path, _ = sample_manifest
        reader = ManifestReader(manifest_path)

        output_path = reader.get_stage_output_path("caption")
        assert "caption" in str(output_path)
        assert "output" in str(output_path)


class TestReplayPlanner:
    """Tests for ReplayPlanner."""

    @pytest.fixture
    def sample_manifest(self, tmp_path: Path) -> tuple[Path, RunManifest]:
        """Create a sample manifest for testing."""
        manifest = RunManifest.create(
            run_id="run_original",
            output_dir=str(tmp_path / "original"),
            run_seed=42,
        )

        manifest = manifest.model_copy(
            update={
                "stages": [
                    StageReference(
                        stage_name="stage_a",
                        stage_type="transform",
                        stage_version="1.0.0",
                        stage_hash="hash_a",
                    ),
                    StageReference(
                        stage_name="stage_b",
                        stage_type="vlm_caption",
                        stage_version="1.0.0",
                        stage_hash="hash_b",
                        depends_on=["stage_a"],
                    ),
                    StageReference(
                        stage_name="stage_c",
                        stage_type="rubric_score",
                        stage_version="1.0.0",
                        stage_hash="hash_c",
                        depends_on=["stage_b"],
                    ),
                ],
            }
        )

        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)

        return manifest_path, manifest

    def test_plan_full_cache_replay(
        self, sample_manifest: tuple[Path, RunManifest], tmp_path: Path
    ) -> None:
        """Test planning full cache replay."""
        manifest_path, _ = sample_manifest
        reader = ManifestReader(manifest_path)
        planner = ReplayPlanner(reader)

        plan = planner.plan_full_cache_replay(
            new_run_id="run_replay",
            output_dir=str(tmp_path / "replay"),
        )

        assert plan.replay_mode == ReplayMode.FULL_CACHE
        assert len(plan.stages) == 3
        assert len(plan.cached_stages) == 3
        assert len(plan.execute_stages) == 0

        # All stages should use cache
        for stage in plan.stages:
            assert stage.use_cache is True

    def test_plan_verify_replay(
        self, sample_manifest: tuple[Path, RunManifest], tmp_path: Path
    ) -> None:
        """Test planning verify replay."""
        manifest_path, _ = sample_manifest
        reader = ManifestReader(manifest_path)
        planner = ReplayPlanner(reader)

        plan = planner.plan_verify_replay(
            new_run_id="run_verify",
            output_dir=str(tmp_path / "verify"),
        )

        assert plan.replay_mode == ReplayMode.VERIFY
        assert len(plan.cached_stages) == 0
        assert len(plan.execute_stages) == 3

        # All stages should execute and verify
        for stage in plan.stages:
            assert stage.use_cache is False
            assert stage.verify_output is True

    def test_plan_from_stage_replay(
        self, sample_manifest: tuple[Path, RunManifest], tmp_path: Path
    ) -> None:
        """Test planning replay from specific stage."""
        manifest_path, _ = sample_manifest
        reader = ManifestReader(manifest_path)
        planner = ReplayPlanner(reader)

        plan = planner.plan_from_stage_replay(
            new_run_id="run_from_b",
            output_dir=str(tmp_path / "from_b"),
            from_stage="stage_b",
        )

        assert plan.replay_mode == ReplayMode.FROM_STAGE
        assert "stage_a" in plan.cached_stages
        assert "stage_b" in plan.execute_stages
        assert "stage_c" in plan.execute_stages

    def test_plan_selective_replay(
        self, sample_manifest: tuple[Path, RunManifest], tmp_path: Path
    ) -> None:
        """Test planning selective replay."""
        manifest_path, _ = sample_manifest
        reader = ManifestReader(manifest_path)
        planner = ReplayPlanner(reader)

        plan = planner.plan_selective_replay(
            new_run_id="run_selective",
            output_dir=str(tmp_path / "selective"),
            stages_to_execute=["stage_b"],
        )

        assert plan.replay_mode == ReplayMode.SELECTIVE
        assert "stage_a" in plan.cached_stages
        assert "stage_b" in plan.execute_stages
        # stage_c depends on stage_b, so it should also execute
        assert "stage_c" in plan.execute_stages

    def test_plan_from_stage_invalid_stage(
        self, sample_manifest: tuple[Path, RunManifest], tmp_path: Path
    ) -> None:
        """Test error when from_stage doesn't exist."""
        manifest_path, _ = sample_manifest
        reader = ManifestReader(manifest_path)
        planner = ReplayPlanner(reader)

        with pytest.raises(ValueError, match="not found"):
            planner.plan_from_stage_replay(
                new_run_id="run_fail",
                output_dir=str(tmp_path / "fail"),
                from_stage="nonexistent",
            )


class TestCreateReplayPlan:
    """Tests for create_replay_plan function."""

    @pytest.fixture
    def sample_manifest(self, tmp_path: Path) -> Path:
        """Create a sample manifest."""
        manifest = RunManifest.create(
            run_id="run_123",
            output_dir=str(tmp_path / "output"),
            run_seed=42,
        )

        manifest = manifest.model_copy(
            update={
                "stages": [
                    StageReference(
                        stage_name="caption",
                        stage_type="vlm_caption",
                        stage_version="1.0.0",
                        stage_hash="abc",
                    ),
                ],
            }
        )

        manifest_path = tmp_path / "manifest.json"
        manifest.save(manifest_path)
        return manifest_path

    def test_create_replay_plan_full_cache(
        self, sample_manifest: Path, tmp_path: Path
    ) -> None:
        """Test creating full cache replay plan."""
        plan = create_replay_plan(
            manifest_path=sample_manifest,
            new_run_id="replay_1",
            output_dir=str(tmp_path / "replay"),
            mode=ReplayMode.FULL_CACHE,
        )

        assert plan.new_run_id == "replay_1"
        assert plan.replay_mode == ReplayMode.FULL_CACHE

    def test_create_replay_plan_verify(
        self, sample_manifest: Path, tmp_path: Path
    ) -> None:
        """Test creating verify replay plan."""
        plan = create_replay_plan(
            manifest_path=sample_manifest,
            new_run_id="verify_1",
            output_dir=str(tmp_path / "verify"),
            mode=ReplayMode.VERIFY,
        )

        assert plan.replay_mode == ReplayMode.VERIFY
        assert plan.should_verify("caption") is True

    def test_create_replay_plan_from_stage_missing_arg(
        self, sample_manifest: Path, tmp_path: Path
    ) -> None:
        """Test error when from_stage missing for FROM_STAGE mode."""
        with pytest.raises(ValueError, match="from_stage required"):
            create_replay_plan(
                manifest_path=sample_manifest,
                new_run_id="fail",
                output_dir=str(tmp_path / "fail"),
                mode=ReplayMode.FROM_STAGE,
            )

    def test_create_replay_plan_selective_missing_arg(
        self, sample_manifest: Path, tmp_path: Path
    ) -> None:
        """Test error when stages_to_execute missing for SELECTIVE mode."""
        with pytest.raises(ValueError, match="stages_to_execute required"):
            create_replay_plan(
                manifest_path=sample_manifest,
                new_run_id="fail",
                output_dir=str(tmp_path / "fail"),
                mode=ReplayMode.SELECTIVE,
            )
