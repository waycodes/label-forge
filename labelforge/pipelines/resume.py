"""
Pipeline resume capability for crash-safe long runs.

Detects partial stage outputs and enables resuming from manifests and cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from labelforge.core.manifest.run_manifest import RunManifest
    from labelforge.core.manifest.stage_manifest import StageManifest


class StageCompletionState(str, Enum):
    """Completion state of a pipeline stage."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    NOT_STARTED = "not_started"


@dataclass
class StageResumeInfo:
    """Information about a stage for resume planning."""

    stage_name: str
    state: StageCompletionState
    completed_rows: int = 0
    total_rows: int = 0
    output_path: str | None = None
    manifest_path: str | None = None
    last_checkpoint: datetime | None = None
    error: str | None = None

    @property
    def completion_ratio(self) -> float:
        """Ratio of completed rows (0.0 to 1.0)."""
        if self.total_rows == 0:
            return 0.0
        return self.completed_rows / self.total_rows

    @property
    def needs_rerun(self) -> bool:
        """Whether this stage needs to be re-run."""
        return self.state in (
            StageCompletionState.PARTIAL,
            StageCompletionState.FAILED,
            StageCompletionState.NOT_STARTED,
        )


@dataclass
class ResumePlan:
    """Plan for resuming a pipeline run."""

    source_run_id: str
    source_manifest_path: str
    stages: list[StageResumeInfo] = field(default_factory=list)
    start_from_stage: str | None = None
    rerun_stages: list[str] = field(default_factory=list)
    skip_stages: list[str] = field(default_factory=list)
    clean_partial: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total_stages(self) -> int:
        """Total number of stages."""
        return len(self.stages)

    @property
    def complete_stages(self) -> int:
        """Number of complete stages."""
        return sum(1 for s in self.stages if s.state == StageCompletionState.COMPLETE)

    @property
    def stages_to_run(self) -> list[str]:
        """List of stage names that need to be run."""
        return self.rerun_stages

    def get_stage_info(self, stage_name: str) -> StageResumeInfo | None:
        """Get info for a specific stage."""
        for stage in self.stages:
            if stage.stage_name == stage_name:
                return stage
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_run_id": self.source_run_id,
            "source_manifest_path": self.source_manifest_path,
            "start_from_stage": self.start_from_stage,
            "rerun_stages": self.rerun_stages,
            "skip_stages": self.skip_stages,
            "total_stages": self.total_stages,
            "complete_stages": self.complete_stages,
            "created_at": self.created_at.isoformat(),
            "stages": [
                {
                    "stage_name": s.stage_name,
                    "state": s.state.value,
                    "completed_rows": s.completed_rows,
                    "total_rows": s.total_rows,
                    "completion_ratio": s.completion_ratio,
                    "needs_rerun": s.needs_rerun,
                    "error": s.error,
                }
                for s in self.stages
            ],
        }


def check_stage_completion(
    stage_output_dir: Path,
    expected_rows: int | None = None,
) -> StageResumeInfo:
    """
    Check the completion state of a stage.

    Args:
        stage_output_dir: Directory containing stage outputs.
        expected_rows: Expected number of output rows (if known).

    Returns:
        StageResumeInfo with completion state.
    """
    stage_name = stage_output_dir.name
    info = StageResumeInfo(stage_name=stage_name)

    if not stage_output_dir.exists():
        info.state = StageCompletionState.NOT_STARTED
        return info

    # Check for manifest
    manifest_path = stage_output_dir / "manifest.json"
    if manifest_path.exists():
        info.manifest_path = str(manifest_path)

        # Try to load and check manifest
        try:
            from labelforge.core.json_canonical import canonical_json_loads

            manifest_data = canonical_json_loads(manifest_path.read_text())
            info.completed_rows = manifest_data.get("output_row_count", 0)
            info.total_rows = manifest_data.get("input_row_count", 0) or expected_rows or 0

            status = manifest_data.get("status", "")
            if status == "complete":
                info.state = StageCompletionState.COMPLETE
            elif status == "failed":
                info.state = StageCompletionState.FAILED
                info.error = manifest_data.get("error")
            else:
                info.state = StageCompletionState.PARTIAL
        except Exception as e:
            info.state = StageCompletionState.PARTIAL
            info.error = f"Failed to load manifest: {e}"
            return info
    else:
        info.state = StageCompletionState.PARTIAL

    # Check for output files
    output_dir = stage_output_dir / "output"
    if output_dir.exists():
        info.output_path = str(output_dir)

        # Count parquet files for row estimation
        parquet_files = list(output_dir.glob("*.parquet"))
        if parquet_files and info.completed_rows == 0:
            # Rough estimate from file count
            info.completed_rows = len(parquet_files) * 100  # Approximate

    # Check for checkpoint
    checkpoint_path = stage_output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        try:
            from labelforge.core.json_canonical import canonical_json_loads

            checkpoint_data = canonical_json_loads(checkpoint_path.read_text())
            checkpoint_time = checkpoint_data.get("timestamp")
            if checkpoint_time:
                info.last_checkpoint = datetime.fromisoformat(checkpoint_time)
        except Exception:
            pass

    return info


def plan_resume(
    run_dir: Path,
    from_stage: str | None = None,
    stages_to_rerun: list[str] | None = None,
    clean_partial: bool = True,
) -> ResumePlan:
    """
    Create a resume plan for a pipeline run.

    Args:
        run_dir: Directory of the run to resume.
        from_stage: Resume from this stage onwards.
        stages_to_rerun: Specific stages to rerun (overrides auto-detection).
        clean_partial: Whether to clean partial outputs before rerun.

    Returns:
        ResumePlan with stages to run.

    Raises:
        FileNotFoundError: If run directory doesn't exist.
        ValueError: If run manifest is invalid.
    """
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Run manifest not found: {manifest_path}")

    # Load run manifest
    from labelforge.core.manifest.run_manifest import RunManifest

    run_manifest = RunManifest.load(manifest_path)
    run_id = run_manifest.metadata.run_id

    # Build resume plan
    plan = ResumePlan(
        source_run_id=run_id,
        source_manifest_path=str(manifest_path),
        start_from_stage=from_stage,
        clean_partial=clean_partial,
    )

    # Check each stage
    stages_dir = run_dir / "stages"
    stage_order = [s.stage_name for s in run_manifest.stages]

    found_from_stage = from_stage is None
    for stage_name in stage_order:
        # Check if we've passed the from_stage
        if from_stage and stage_name == from_stage:
            found_from_stage = True

        stage_dir = stages_dir / stage_name
        info = check_stage_completion(stage_dir)
        plan.stages.append(info)

        # Determine if stage needs to run
        if stages_to_rerun and stage_name in stages_to_rerun:
            plan.rerun_stages.append(stage_name)
        elif found_from_stage and info.needs_rerun:
            plan.rerun_stages.append(stage_name)
        elif found_from_stage and from_stage:
            # After from_stage, include all stages
            plan.rerun_stages.append(stage_name)
        else:
            plan.skip_stages.append(stage_name)

    return plan


def clean_partial_stage(stage_dir: Path) -> bool:
    """
    Clean up partial stage outputs for a clean rerun.

    Args:
        stage_dir: Directory of the stage to clean.

    Returns:
        True if cleanup was performed.
    """
    import shutil

    if not stage_dir.exists():
        return False

    cleaned = False

    # Remove output directory
    output_dir = stage_dir / "output"
    if output_dir.exists():
        shutil.rmtree(output_dir)
        cleaned = True

    # Remove row records (but keep manifest for history)
    rows_file = stage_dir / "rows.jsonl"
    if rows_file.exists():
        rows_file.unlink()
        cleaned = True

    # Remove checkpoint
    checkpoint_file = stage_dir / "checkpoint.json"
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        cleaned = True

    return cleaned


class CheckpointManager:
    """
    Manages checkpoints for crash recovery during stage execution.
    """

    def __init__(self, stage_dir: Path, checkpoint_interval: int = 100):
        """
        Initialize checkpoint manager.

        Args:
            stage_dir: Directory for stage outputs.
            checkpoint_interval: Save checkpoint every N rows.
        """
        self.stage_dir = stage_dir
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_path = stage_dir / "checkpoint.json"
        self.rows_processed = 0
        self.last_row_id: str | None = None
        self.last_checkpoint_time: datetime | None = None

    def should_checkpoint(self) -> bool:
        """Check if a checkpoint should be saved."""
        return self.rows_processed % self.checkpoint_interval == 0

    def save_checkpoint(
        self,
        row_id: str,
        additional_state: dict[str, Any] | None = None,
    ) -> None:
        """
        Save a checkpoint.

        Args:
            row_id: ID of the last processed row.
            additional_state: Additional state to save.
        """
        from labelforge.core.json_canonical import canonical_json_dumps

        self.last_row_id = row_id
        self.last_checkpoint_time = datetime.utcnow()

        checkpoint_data = {
            "timestamp": self.last_checkpoint_time.isoformat(),
            "rows_processed": self.rows_processed,
            "last_row_id": row_id,
        }

        if additional_state:
            checkpoint_data["state"] = additional_state

        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.write_text(canonical_json_dumps(checkpoint_data, indent=True))

    def load_checkpoint(self) -> dict[str, Any] | None:
        """
        Load the last checkpoint if it exists.

        Returns:
            Checkpoint data or None if no checkpoint.
        """
        if not self.checkpoint_path.exists():
            return None

        try:
            from labelforge.core.json_canonical import canonical_json_loads

            data = canonical_json_loads(self.checkpoint_path.read_text())
            self.rows_processed = data.get("rows_processed", 0)
            self.last_row_id = data.get("last_row_id")
            if data.get("timestamp"):
                self.last_checkpoint_time = datetime.fromisoformat(data["timestamp"])
            return data
        except Exception:
            return None

    def record_row(self, row_id: str) -> None:
        """
        Record that a row was processed.

        Args:
            row_id: ID of the processed row.
        """
        self.rows_processed += 1

        if self.should_checkpoint():
            self.save_checkpoint(row_id)

    def clear(self) -> None:
        """Clear the checkpoint file."""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()


@dataclass
class ResumeContext:
    """
    Context for stage execution with resume support.
    """

    plan: ResumePlan
    current_stage: str
    checkpoint_manager: CheckpointManager | None = None
    skip_until_row: str | None = None  # Resume from this row ID

    @property
    def is_resuming(self) -> bool:
        """Whether we're resuming a previous run."""
        return self.skip_until_row is not None

    @property
    def stage_info(self) -> StageResumeInfo | None:
        """Get info for current stage."""
        return self.plan.get_stage_info(self.current_stage)

    def should_skip_row(self, row_id: str) -> bool:
        """
        Check if a row should be skipped (already processed).

        Args:
            row_id: Row ID to check.

        Returns:
            True if row should be skipped.
        """
        if not self.skip_until_row:
            return False

        if row_id == self.skip_until_row:
            # Found the resume point, stop skipping after this
            self.skip_until_row = None
            return True  # Skip this one too since it was partially processed

        return True  # Keep skipping

    def mark_complete(self, row_id: str) -> None:
        """
        Mark a row as complete.

        Args:
            row_id: Completed row ID.
        """
        if self.checkpoint_manager:
            self.checkpoint_manager.record_row(row_id)
