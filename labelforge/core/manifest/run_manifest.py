"""
Run manifest schema.

Run-level metadata that enables full replay of a pipeline execution.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from labelforge.core.json_canonical import canonical_json_dumps


class RunMetadata(BaseModel):
    """Metadata for a pipeline run."""

    model_config = ConfigDict(frozen=True)

    # Identity
    run_id: str = Field(description="Unique run identifier")
    run_name: str | None = Field(default=None, description="Human-readable run name")

    # Timing
    started_at: datetime = Field(description="Run start timestamp")
    completed_at: datetime | None = Field(
        default=None, description="Run completion timestamp"
    )

    # Source info
    git_commit: str | None = Field(default=None, description="Git commit hash")
    git_branch: str | None = Field(default=None, description="Git branch name")
    git_dirty: bool = Field(default=False, description="Whether working tree was dirty")

    # Environment
    env_snapshot_path: str | None = Field(
        default=None, description="Path to environment snapshot JSON"
    )

    # Configuration
    config_path: str | None = Field(
        default=None, description="Path to pipeline config"
    )
    config_hash: str | None = Field(default=None, description="Hash of pipeline config")

    # Seeds
    run_seed: int = Field(description="Base seed for the run")


class StageReference(BaseModel):
    """Reference to a stage in the pipeline."""

    model_config = ConfigDict(frozen=True)

    stage_name: str = Field(description="Stage name")
    stage_type: str = Field(description="Stage type (e.g., vlm_caption)")
    stage_version: str = Field(description="Stage version")
    stage_hash: str = Field(description="Stage configuration hash")
    depends_on: list[str] = Field(
        default_factory=list, description="Names of upstream stages"
    )


class RunManifest(BaseModel):
    """
    Complete manifest for a pipeline run.

    Contains all information needed to replay the run.
    """

    model_config = ConfigDict(frozen=True)

    # Run metadata
    metadata: RunMetadata = Field(description="Run-level metadata")

    # Prompt packs used
    prompt_packs: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Prompt pack fingerprints by name",
    )

    # Model specs used
    model_specs: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Model spec fingerprints by name",
    )

    # Stage DAG
    stages: list[StageReference] = Field(
        default_factory=list, description="Ordered list of stages"
    )

    # Stage manifest paths
    stage_manifests: dict[str, str] = Field(
        default_factory=dict,
        description="Paths to stage manifests by stage name",
    )

    # Output paths
    output_dir: str = Field(description="Root output directory for this run")

    # Run status
    status: str = Field(default="running", description="Run status")
    error: str | None = Field(default=None, description="Error message if failed")

    # Manifest hash (computed on finalization)
    manifest_hash: str | None = Field(
        default=None, description="Hash of finalized manifest"
    )

    def to_json(self, indent: bool = True) -> str:
        """Serialize to canonical JSON."""
        return canonical_json_dumps(self.model_dump(), indent=indent)

    def save(self, path: Path) -> None:
        """Save manifest to file."""
        path.write_text(self.to_json(indent=True))

    @classmethod
    def load(cls, path: Path) -> RunManifest:
        """Load manifest from file."""
        from labelforge.core.json_canonical import canonical_json_loads

        content = path.read_text()
        data = canonical_json_loads(content)
        return cls.model_validate(data)

    @classmethod
    def create(
        cls,
        run_id: str,
        output_dir: str,
        run_seed: int,
        run_name: str | None = None,
        git_commit: str | None = None,
        git_branch: str | None = None,
        git_dirty: bool = False,
        env_snapshot_path: str | None = None,
        config_path: str | None = None,
        config_hash: str | None = None,
    ) -> RunManifest:
        """
        Create a new run manifest.

        Args:
            run_id: Unique run identifier.
            output_dir: Root output directory.
            run_seed: Base seed for the run.
            run_name: Optional human-readable name.
            git_commit: Optional git commit hash.
            git_branch: Optional git branch.
            git_dirty: Whether working tree is dirty.
            env_snapshot_path: Path to environment snapshot.
            config_path: Path to pipeline config.
            config_hash: Hash of pipeline config.

        Returns:
            New RunManifest instance.
        """
        metadata = RunMetadata(
            run_id=run_id,
            run_name=run_name,
            started_at=datetime.utcnow(),
            git_commit=git_commit,
            git_branch=git_branch,
            git_dirty=git_dirty,
            env_snapshot_path=env_snapshot_path,
            config_path=config_path,
            config_hash=config_hash,
            run_seed=run_seed,
        )

        return cls(metadata=metadata, output_dir=output_dir)
