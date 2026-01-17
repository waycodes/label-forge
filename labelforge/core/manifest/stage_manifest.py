"""
Stage manifest schema.

Stage-level accounting enables partial replay and caching.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from labelforge.core.json_canonical import canonical_json_dumps


class CacheStats(BaseModel):
    """Statistics about cache usage for a stage."""

    model_config = ConfigDict(frozen=True)

    hits: int = Field(default=0, description="Number of cache hits")
    misses: int = Field(default=0, description="Number of cache misses")
    bytes_read: int = Field(default=0, description="Bytes read from cache")
    bytes_written: int = Field(default=0, description="Bytes written to cache")
    hit_rate: float = Field(default=0.0, description="Cache hit rate")

    @classmethod
    def compute(cls, hits: int, misses: int, bytes_read: int, bytes_written: int) -> CacheStats:
        """Compute cache stats including hit rate."""
        total = hits + misses
        hit_rate = hits / total if total > 0 else 0.0
        return cls(
            hits=hits,
            misses=misses,
            bytes_read=bytes_read,
            bytes_written=bytes_written,
            hit_rate=hit_rate,
        )


class TimingStats(BaseModel):
    """Timing statistics for a stage."""

    model_config = ConfigDict(frozen=True)

    total_ms: float = Field(default=0.0, description="Total stage time in ms")
    preprocess_ms: float = Field(default=0.0, description="Preprocessing time in ms")
    inference_ms: float = Field(default=0.0, description="Model inference time in ms")
    postprocess_ms: float = Field(default=0.0, description="Postprocessing time in ms")
    write_ms: float = Field(default=0.0, description="Output writing time in ms")


class StageManifest(BaseModel):
    """
    Complete manifest for a pipeline stage.

    Contains all information needed for partial replay.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    stage_name: str = Field(description="Stage name")
    stage_type: str = Field(description="Stage type")
    stage_version: str = Field(description="Stage version")

    # Run context
    run_id: str = Field(description="Parent run ID")
    stage_index: int = Field(description="Stage index in pipeline")

    # Timing
    started_at: datetime = Field(description="Stage start timestamp")
    completed_at: datetime | None = Field(
        default=None, description="Stage completion timestamp"
    )

    # Input fingerprints
    input_dataset_path: str | None = Field(
        default=None, description="Path to input dataset"
    )
    input_dataset_hash: str | None = Field(
        default=None, description="Hash of input dataset"
    )
    input_row_count: int = Field(default=0, description="Number of input rows")

    # Config fingerprints
    stage_config_hash: str = Field(description="Hash of stage configuration")
    prompt_hash: str | None = Field(default=None, description="Hash of prompt used")
    model_hash: str | None = Field(default=None, description="Hash of model used")

    # Output fingerprints
    output_dataset_path: str = Field(description="Path to output dataset")
    output_dataset_hash: str | None = Field(
        default=None, description="Hash of output dataset"
    )
    output_row_count: int = Field(default=0, description="Number of output rows")
    output_schema_hash: str | None = Field(
        default=None, description="Hash of output schema"
    )

    # Row manifest path
    row_manifest_path: str | None = Field(
        default=None, description="Path to row-level manifest (JSONL)"
    )

    # Statistics
    cache_stats: CacheStats = Field(
        default_factory=CacheStats, description="Cache usage statistics"
    )
    timing_stats: TimingStats = Field(
        default_factory=TimingStats, description="Timing statistics"
    )

    # Error handling
    error_count: int = Field(default=0, description="Number of row-level errors")
    status: str = Field(default="running", description="Stage status")
    error: str | None = Field(default=None, description="Stage-level error message")

    def to_json(self, indent: bool = True) -> str:
        """Serialize to canonical JSON."""
        return canonical_json_dumps(self.model_dump(), indent=indent)

    def save(self, path: Path) -> None:
        """Save manifest to file."""
        path.write_text(self.to_json(indent=True))

    @classmethod
    def load(cls, path: Path) -> StageManifest:
        """Load manifest from file."""
        from labelforge.core.json_canonical import canonical_json_loads

        content = path.read_text()
        data = canonical_json_loads(content)
        return cls.model_validate(data)

    @classmethod
    def create(
        cls,
        stage_name: str,
        stage_type: str,
        stage_version: str,
        run_id: str,
        stage_index: int,
        stage_config_hash: str,
        output_dataset_path: str,
        input_dataset_path: str | None = None,
        input_dataset_hash: str | None = None,
        input_row_count: int = 0,
        prompt_hash: str | None = None,
        model_hash: str | None = None,
    ) -> StageManifest:
        """
        Create a new stage manifest.

        Args:
            stage_name: Stage name.
            stage_type: Stage type.
            stage_version: Stage version.
            run_id: Parent run ID.
            stage_index: Index in pipeline.
            stage_config_hash: Hash of stage config.
            output_dataset_path: Path to output dataset.
            input_dataset_path: Optional path to input dataset.
            input_dataset_hash: Optional hash of input dataset.
            input_row_count: Number of input rows.
            prompt_hash: Optional hash of prompt.
            model_hash: Optional hash of model.

        Returns:
            New StageManifest instance.
        """
        return cls(
            stage_name=stage_name,
            stage_type=stage_type,
            stage_version=stage_version,
            run_id=run_id,
            stage_index=stage_index,
            started_at=datetime.utcnow(),
            input_dataset_path=input_dataset_path,
            input_dataset_hash=input_dataset_hash,
            input_row_count=input_row_count,
            stage_config_hash=stage_config_hash,
            prompt_hash=prompt_hash,
            model_hash=model_hash,
            output_dataset_path=output_dataset_path,
        )
