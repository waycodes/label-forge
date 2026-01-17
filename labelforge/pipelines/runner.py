"""
Pipeline runner.

Orchestrates stages with caching, manifests, and deterministic execution.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import ray.data

from labelforge.core.env_capture import capture_environment
from labelforge.core.manifest.run_manifest import RunManifest, StageReference
from labelforge.core.manifest.stage_manifest import StageManifest, CacheStats
from labelforge.core.seeds import SeedPolicy
from labelforge.pipelines.dag import PipelineDAG
from labelforge.pipelines.stage import Stage, StageConfig, StageContext, StageResult


@dataclass
class RunConfig:
    """Configuration for a pipeline run."""

    # Identity
    run_name: str | None = None

    # Output
    output_dir: str = "runs"

    # Determinism
    seed: int = 42
    deterministic_mode: bool = True

    # Caching
    cache_enabled: bool = True
    cache_dir: str | None = None

    # Execution
    max_retries: int = 3
    preserve_order: bool = False


@dataclass
class PipelineRunner:
    """
    Orchestrates pipeline execution with caching and manifests.

    Executes stages in topological order, managing caching, manifests,
    and error handling.
    """

    dag: PipelineDAG
    stages: dict[str, Stage] = field(default_factory=dict)
    config: RunConfig = field(default_factory=RunConfig)

    # Runtime state
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    run_manifest: RunManifest | None = None

    def register_stage(self, stage: Stage) -> None:
        """
        Register a stage implementation.

        Args:
            stage: Stage instance to register.
        """
        self.stages[stage.name] = stage

    def prepare_run(self) -> RunManifest:
        """
        Prepare for a pipeline run.

        Creates output directories, captures environment, and initializes manifest.

        Returns:
            Initialized RunManifest.
        """
        # Create run directory
        run_dir = Path(self.config.output_dir) / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Capture environment
        env_snapshot = capture_environment()
        env_path = run_dir / "env_snapshot.json"
        env_snapshot.save(env_path)

        # Create manifest
        self.run_manifest = RunManifest.create(
            run_id=self.run_id,
            output_dir=str(run_dir),
            run_seed=self.config.seed,
            run_name=self.config.run_name,
            git_commit=env_snapshot.git_commit,
            git_branch=env_snapshot.git_branch,
            git_dirty=env_snapshot.git_dirty,
            env_snapshot_path=str(env_path),
        )

        # Add stage references to manifest
        stage_refs = []
        for node_name in self.dag.get_execution_order():
            node = self.dag.get_node(node_name)
            if node:
                stage = self.stages.get(node_name)
                stage_refs.append(
                    StageReference(
                        stage_name=node_name,
                        stage_type=node.stage_type,
                        stage_version=stage.version if stage else "unknown",
                        stage_hash=stage.fingerprint() if stage else "",
                        depends_on=node.depends_on,
                    )
                )

        # Update manifest with stages
        self.run_manifest = RunManifest(
            metadata=self.run_manifest.metadata,
            stages=stage_refs,
            output_dir=str(run_dir),
        )

        return self.run_manifest

    def run(
        self,
        initial_dataset: ray.data.Dataset,
    ) -> dict[str, ray.data.Dataset]:
        """
        Execute the full pipeline.

        Args:
            initial_dataset: Input dataset for the first stage.

        Returns:
            Dict mapping stage names to output datasets.
        """
        # Prepare run
        if self.run_manifest is None:
            self.prepare_run()

        # Initialize seed policy
        seed_policy = SeedPolicy(run_seed=self.config.seed)

        # Get execution order
        execution_order = self.dag.get_execution_order()

        # Track datasets by stage name
        datasets: dict[str, ray.data.Dataset] = {}

        # Execute stages in order
        for i, stage_name in enumerate(execution_order):
            stage = self.stages.get(stage_name)
            if stage is None:
                raise ValueError(f"No implementation registered for stage '{stage_name}'")

            node = self.dag.get_node(stage_name)
            if node is None:
                raise ValueError(f"Node '{stage_name}' not found in DAG")

            # Get input dataset
            if node.depends_on:
                # Use output from dependency (for now, just use first)
                input_ds = datasets[node.depends_on[0]]
            else:
                input_ds = initial_dataset

            # Create stage context
            run_dir = Path(self.run_manifest.output_dir) if self.run_manifest else Path(".")
            stage_output_dir = run_dir / "stages" / stage_name
            stage_output_dir.mkdir(parents=True, exist_ok=True)

            context = StageContext(
                run_id=self.run_id,
                stage_index=i,
                output_dir=str(stage_output_dir),
                stage_seed=seed_policy.derive_stage_seed(stage_name),
                cache_enabled=self.config.cache_enabled,
            )

            # Validate input
            validation_errors = stage.validate_input(input_ds)
            if validation_errors:
                raise ValueError(
                    f"Stage '{stage_name}' input validation failed: {validation_errors}"
                )

            # Execute stage
            output_ds = stage.run(input_ds, context)

            # Store output
            datasets[stage_name] = output_ds

        return datasets

    def get_run_dir(self) -> Path:
        """Get the run output directory."""
        return Path(self.config.output_dir) / self.run_id


def create_runner(
    dag: PipelineDAG,
    config: RunConfig | None = None,
) -> PipelineRunner:
    """
    Create a pipeline runner.

    Args:
        dag: Pipeline DAG definition.
        config: Optional run configuration.

    Returns:
        Configured PipelineRunner.
    """
    return PipelineRunner(
        dag=dag,
        config=config or RunConfig(),
    )
