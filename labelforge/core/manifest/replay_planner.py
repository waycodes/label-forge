"""
Manifest Reader and Replay Planner.

Reads run manifests and generates plans for deterministic replay.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from labelforge.core.manifest.run_manifest import RunManifest, StageReference


class ReplayMode(str, Enum):
    """Mode for replay execution."""

    # Fully deterministic replay using cached outputs
    FULL_CACHE = "full_cache"
    # Re-execute all stages, verify outputs match
    VERIFY = "verify"
    # Re-execute from a specific stage
    FROM_STAGE = "from_stage"
    # Re-execute only specified stages
    SELECTIVE = "selective"


@dataclass
class StageReplayTask:
    """A single stage to replay."""

    stage_name: str
    stage_type: str
    stage_hash: str

    # Replay behavior
    use_cache: bool = True
    verify_output: bool = False

    # Dependencies
    depends_on: list[str] = field(default_factory=list)

    # Paths
    cached_output_path: str | None = None
    output_path: str | None = None


@dataclass
class ReplayPlan:
    """
    Plan for replaying a pipeline run.

    Contains ordered list of stage tasks with dependency info.
    """

    # Source manifest
    source_manifest_path: str
    source_run_id: str
    source_run_seed: int

    # Replay settings
    replay_mode: ReplayMode
    new_run_id: str
    output_dir: str

    # Stage tasks in execution order
    stages: list[StageReplayTask] = field(default_factory=list)

    # Stages to skip (use cached output)
    cached_stages: set[str] = field(default_factory=set)

    # Stages to re-execute
    execute_stages: set[str] = field(default_factory=set)

    # Created timestamp
    created_at: datetime = field(default_factory=datetime.utcnow)

    def get_stage_order(self) -> list[str]:
        """Get ordered list of stage names."""
        return [s.stage_name for s in self.stages]

    def should_execute(self, stage_name: str) -> bool:
        """Check if a stage should be executed (not cached)."""
        return stage_name in self.execute_stages

    def should_verify(self, stage_name: str) -> bool:
        """Check if a stage output should be verified."""
        if self.replay_mode == ReplayMode.VERIFY:
            return True
        return any(s.stage_name == stage_name and s.verify_output for s in self.stages)


class ManifestReader:
    """
    Reader for run manifests and associated data.

    Provides access to run metadata, stage info, and cached outputs.
    """

    def __init__(self, manifest_path: Path | str):
        """
        Initialize manifest reader.

        Args:
            manifest_path: Path to run manifest file.
        """
        self.manifest_path = Path(manifest_path)
        self._manifest: RunManifest | None = None
        self._stage_manifests: dict[str, Any] = {}

    @property
    def manifest(self) -> RunManifest:
        """Get loaded manifest."""
        if self._manifest is None:
            self._manifest = self._load_manifest()
        return self._manifest

    def _load_manifest(self) -> RunManifest:
        """Load manifest from file."""
        from labelforge.core.manifest.run_manifest import RunManifest

        return RunManifest.load(self.manifest_path)

    @property
    def run_id(self) -> str:
        """Get run ID."""
        return self.manifest.metadata.run_id

    @property
    def run_seed(self) -> int:
        """Get run seed."""
        return self.manifest.metadata.run_seed

    @property
    def output_dir(self) -> str:
        """Get output directory."""
        return self.manifest.output_dir

    @property
    def stages(self) -> list[StageReference]:
        """Get list of stages."""
        return self.manifest.stages

    def get_stage(self, name: str) -> StageReference | None:
        """Get stage reference by name."""
        for stage in self.stages:
            if stage.stage_name == name:
                return stage
        return None

    def get_stage_manifest_path(self, stage_name: str) -> Path | None:
        """Get path to stage manifest file."""
        path_str = self.manifest.stage_manifests.get(stage_name)
        if path_str:
            return Path(path_str)
        return None

    def get_stage_output_path(self, stage_name: str) -> Path:
        """Get expected output path for a stage."""
        return Path(self.output_dir) / stage_name / "output"

    def get_prompt_pack_hash(self, name: str) -> str | None:
        """Get prompt pack hash by name."""
        pack_info = self.manifest.prompt_packs.get(name, {})
        return pack_info.get("hash")

    def get_model_spec_hash(self, name: str) -> str | None:
        """Get model spec hash by name."""
        spec_info = self.manifest.model_specs.get(name, {})
        return spec_info.get("hash")

    def load_stage_manifest(self, stage_name: str) -> dict[str, Any] | None:
        """Load stage manifest for detailed row-level info."""
        if stage_name in self._stage_manifests:
            return self._stage_manifests[stage_name]

        path = self.get_stage_manifest_path(stage_name)
        if path is None or not path.exists():
            return None

        from labelforge.io.dataset_rw import read_jsonl_manifest

        # Stage manifests are JSONL files
        records = read_jsonl_manifest(path)
        self._stage_manifests[stage_name] = {"records": records}
        return self._stage_manifests[stage_name]

    def validate_manifest(self) -> list[str]:
        """
        Validate manifest consistency.

        Returns:
            List of validation errors.
        """
        errors = []

        # Check required fields
        if not self.manifest.metadata.run_id:
            errors.append("Missing run_id in metadata")

        if not self.manifest.output_dir:
            errors.append("Missing output_dir")

        # Check stage dependencies exist
        stage_names = {s.stage_name for s in self.stages}
        for stage in self.stages:
            for dep in stage.depends_on:
                if dep not in stage_names:
                    errors.append(
                        f"Stage '{stage.stage_name}' depends on "
                        f"unknown stage '{dep}'"
                    )

        return errors


class ReplayPlanner:
    """
    Plans replay execution from a run manifest.

    Analyzes the manifest and determines which stages to re-execute
    vs retrieve from cache.
    """

    def __init__(self, manifest_reader: ManifestReader):
        """
        Initialize replay planner.

        Args:
            manifest_reader: Reader for source manifest.
        """
        self.reader = manifest_reader

    def plan_full_cache_replay(
        self,
        new_run_id: str,
        output_dir: str,
    ) -> ReplayPlan:
        """
        Plan a replay using all cached outputs.

        Args:
            new_run_id: Run ID for the replay.
            output_dir: Output directory for replay.

        Returns:
            Replay plan with all stages cached.
        """
        plan = ReplayPlan(
            source_manifest_path=str(self.reader.manifest_path),
            source_run_id=self.reader.run_id,
            source_run_seed=self.reader.run_seed,
            replay_mode=ReplayMode.FULL_CACHE,
            new_run_id=new_run_id,
            output_dir=output_dir,
        )

        for stage in self.reader.stages:
            task = StageReplayTask(
                stage_name=stage.stage_name,
                stage_type=stage.stage_type,
                stage_hash=stage.stage_hash,
                use_cache=True,
                verify_output=False,
                depends_on=list(stage.depends_on),
                cached_output_path=str(
                    self.reader.get_stage_output_path(stage.stage_name)
                ),
            )
            plan.stages.append(task)
            plan.cached_stages.add(stage.stage_name)

        return plan

    def plan_verify_replay(
        self,
        new_run_id: str,
        output_dir: str,
    ) -> ReplayPlan:
        """
        Plan a replay that re-executes and verifies all stages.

        Args:
            new_run_id: Run ID for the replay.
            output_dir: Output directory for replay.

        Returns:
            Replay plan with all stages to execute and verify.
        """
        plan = ReplayPlan(
            source_manifest_path=str(self.reader.manifest_path),
            source_run_id=self.reader.run_id,
            source_run_seed=self.reader.run_seed,
            replay_mode=ReplayMode.VERIFY,
            new_run_id=new_run_id,
            output_dir=output_dir,
        )

        for stage in self.reader.stages:
            task = StageReplayTask(
                stage_name=stage.stage_name,
                stage_type=stage.stage_type,
                stage_hash=stage.stage_hash,
                use_cache=False,
                verify_output=True,
                depends_on=list(stage.depends_on),
                cached_output_path=str(
                    self.reader.get_stage_output_path(stage.stage_name)
                ),
                output_path=f"{output_dir}/{stage.stage_name}/output",
            )
            plan.stages.append(task)
            plan.execute_stages.add(stage.stage_name)

        return plan

    def plan_from_stage_replay(
        self,
        new_run_id: str,
        output_dir: str,
        from_stage: str,
    ) -> ReplayPlan:
        """
        Plan a replay starting from a specific stage.

        Stages before the specified stage use cached outputs.

        Args:
            new_run_id: Run ID for the replay.
            output_dir: Output directory for replay.
            from_stage: Stage name to start execution from.

        Returns:
            Replay plan with stages before from_stage cached.
        """
        plan = ReplayPlan(
            source_manifest_path=str(self.reader.manifest_path),
            source_run_id=self.reader.run_id,
            source_run_seed=self.reader.run_seed,
            replay_mode=ReplayMode.FROM_STAGE,
            new_run_id=new_run_id,
            output_dir=output_dir,
        )

        # Find the index of the from_stage
        stage_names = [s.stage_name for s in self.reader.stages]
        if from_stage not in stage_names:
            raise ValueError(f"Stage '{from_stage}' not found in manifest")

        from_index = stage_names.index(from_stage)

        for i, stage in enumerate(self.reader.stages):
            use_cache = i < from_index

            task = StageReplayTask(
                stage_name=stage.stage_name,
                stage_type=stage.stage_type,
                stage_hash=stage.stage_hash,
                use_cache=use_cache,
                verify_output=False,
                depends_on=list(stage.depends_on),
                cached_output_path=str(
                    self.reader.get_stage_output_path(stage.stage_name)
                ),
                output_path=f"{output_dir}/{stage.stage_name}/output"
                if not use_cache
                else None,
            )
            plan.stages.append(task)

            if use_cache:
                plan.cached_stages.add(stage.stage_name)
            else:
                plan.execute_stages.add(stage.stage_name)

        return plan

    def plan_selective_replay(
        self,
        new_run_id: str,
        output_dir: str,
        stages_to_execute: list[str],
    ) -> ReplayPlan:
        """
        Plan a replay executing only selected stages.

        Args:
            new_run_id: Run ID for the replay.
            output_dir: Output directory for replay.
            stages_to_execute: List of stage names to re-execute.

        Returns:
            Replay plan with selected stages to execute.
        """
        plan = ReplayPlan(
            source_manifest_path=str(self.reader.manifest_path),
            source_run_id=self.reader.run_id,
            source_run_seed=self.reader.run_seed,
            replay_mode=ReplayMode.SELECTIVE,
            new_run_id=new_run_id,
            output_dir=output_dir,
        )

        execute_set = set(stages_to_execute)

        # Also include all downstream dependencies
        stage_names = [s.stage_name for s in self.reader.stages]
        for stage_name in stages_to_execute:
            if stage_name not in stage_names:
                raise ValueError(f"Stage '{stage_name}' not found in manifest")

            # Find downstream stages
            idx = stage_names.index(stage_name)
            for downstream_stage in self.reader.stages[idx + 1 :]:
                if any(dep in execute_set for dep in downstream_stage.depends_on):
                    execute_set.add(downstream_stage.stage_name)

        for stage in self.reader.stages:
            use_cache = stage.stage_name not in execute_set

            task = StageReplayTask(
                stage_name=stage.stage_name,
                stage_type=stage.stage_type,
                stage_hash=stage.stage_hash,
                use_cache=use_cache,
                verify_output=False,
                depends_on=list(stage.depends_on),
                cached_output_path=str(
                    self.reader.get_stage_output_path(stage.stage_name)
                ),
                output_path=f"{output_dir}/{stage.stage_name}/output"
                if not use_cache
                else None,
            )
            plan.stages.append(task)

            if use_cache:
                plan.cached_stages.add(stage.stage_name)
            else:
                plan.execute_stages.add(stage.stage_name)

        return plan


def create_replay_plan(
    manifest_path: str | Path,
    new_run_id: str,
    output_dir: str,
    *,
    mode: ReplayMode = ReplayMode.FULL_CACHE,
    from_stage: str | None = None,
    stages_to_execute: list[str] | None = None,
) -> ReplayPlan:
    """
    Create a replay plan from a manifest.

    Args:
        manifest_path: Path to source run manifest.
        new_run_id: Run ID for the replay.
        output_dir: Output directory for replay.
        mode: Replay mode.
        from_stage: Stage to start from (for FROM_STAGE mode).
        stages_to_execute: Stages to execute (for SELECTIVE mode).

    Returns:
        Replay plan.

    Example:
        >>> plan = create_replay_plan(
        ...     "runs/run_123/manifest.json",
        ...     new_run_id="run_456",
        ...     output_dir="runs/run_456",
        ...     mode=ReplayMode.VERIFY,
        ... )
    """
    reader = ManifestReader(manifest_path)

    # Validate manifest
    errors = reader.validate_manifest()
    if errors:
        raise ValueError(f"Invalid manifest: {errors}")

    planner = ReplayPlanner(reader)

    if mode == ReplayMode.FULL_CACHE:
        return planner.plan_full_cache_replay(new_run_id, output_dir)
    elif mode == ReplayMode.VERIFY:
        return planner.plan_verify_replay(new_run_id, output_dir)
    elif mode == ReplayMode.FROM_STAGE:
        if from_stage is None:
            raise ValueError("from_stage required for FROM_STAGE mode")
        return planner.plan_from_stage_replay(new_run_id, output_dir, from_stage)
    elif mode == ReplayMode.SELECTIVE:
        if stages_to_execute is None:
            raise ValueError("stages_to_execute required for SELECTIVE mode")
        return planner.plan_selective_replay(new_run_id, output_dir, stages_to_execute)
    else:
        raise ValueError(f"Unknown replay mode: {mode}")
