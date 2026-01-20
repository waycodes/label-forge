"""
LabelForge CLI.

Command-line interface for running pipelines, replays, and inspections.
"""

import click
import yaml

from labelforge import __version__


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """LabelForge: Deterministic multimodal labeling pipeline."""
    pass


@main.command()
@click.option("--config", "-c", required=True, help="Path to pipeline config YAML")
@click.option("--output", "-o", default="runs", help="Output directory")
@click.option("--seed", default=42, type=int, help="Random seed")
@click.option("--name", "-n", help="Run name")
@click.option("--deterministic/--no-deterministic", default=True, help="Enable determinism")
@click.option("--cache/--no-cache", default=True, help="Enable caching")
@click.option("--dry-run", is_flag=True, help="Validate config without executing")
def run(
    config: str,
    output: str,
    seed: int,
    name: str | None,
    deterministic: bool,
    cache: bool,
    dry_run: bool,
) -> None:
    """Run a labeling pipeline."""
    from pathlib import Path

    from labelforge.pipelines.dag import PipelineDAG
    from labelforge.pipelines.runner import RunConfig, create_runner

    click.echo(f"Loading pipeline config: {config}")

    # Load config
    config_path = Path(config)
    if not config_path.exists():
        click.echo(f"Error: Config file not found: {config}", err=True)
        raise SystemExit(1)

    try:
        with config_path.open() as f:
            pipeline_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        click.echo(f"Error parsing config: {e}", err=True)
        raise SystemExit(1)

    # Create DAG from config
    dag = PipelineDAG()
    stages_config = pipeline_config.get("stages", [])

    for stage_cfg in stages_config:
        stage_name = stage_cfg.get("name")
        stage_type = stage_cfg.get("type", "transform")
        depends_on = stage_cfg.get("depends_on", [])

        if not stage_name:
            click.echo("Error: Stage missing 'name' field", err=True)
            raise SystemExit(1)

        dag.add_node(stage_name, stage_type=stage_type, depends_on=depends_on)

    click.echo(f"Pipeline has {len(stages_config)} stages")

    if dry_run:
        click.echo("\nDry run - validating configuration...")
        execution_order = dag.get_execution_order()
        click.echo(f"Execution order: {' -> '.join(execution_order)}")
        click.echo("Configuration is valid!")
        return

    # Create runner
    run_config = RunConfig(
        run_name=name,
        output_dir=output,
        seed=seed,
        deterministic_mode=deterministic,
        cache_enabled=cache,
    )

    runner = create_runner(dag, run_config)

    click.echo(f"\nStarting run: {runner.run_id}")
    click.echo(f"Output directory: {output}/{runner.run_id}")
    click.echo(f"Seed: {seed}")
    click.echo(f"Deterministic: {deterministic}")
    click.echo(f"Cache enabled: {cache}")

    # Note: Actual stage registration and dataset loading
    # would happen here, connecting stages to their implementations
    click.echo("\nPipeline prepared. Stage registration requires implementation bindings.")
    click.echo("Use the Python API for full pipeline execution.")


@main.command()
@click.argument("manifest_path")
@click.option("--output", "-o", help="Output directory (defaults to new run dir)")
@click.option("--mode", type=click.Choice(["cache", "verify", "from-stage", "selective"]),
              default="cache", help="Replay mode")
@click.option("--from-stage", help="Stage to start from (for from-stage mode)")
@click.option("--stages", help="Comma-separated stages to execute (for selective mode)")
@click.option("--validate/--no-validate", default=True, help="Validate environment match")
def replay(
    manifest_path: str,
    output: str | None,
    mode: str,
    from_stage: str | None,
    stages: str | None,
    validate: bool,
) -> None:
    """Replay a previous run from its manifest."""
    from pathlib import Path

    from labelforge.core.manifest.replay_planner import (
        ManifestReader,
        ReplayMode,
        create_replay_plan,
    )

    click.echo(f"Loading manifest: {manifest_path}")

    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        click.echo(f"Error: Manifest not found: {manifest_path}", err=True)
        raise SystemExit(1)

    # Load and validate manifest
    try:
        reader = ManifestReader(manifest_file)
        errors = reader.validate_manifest()
        if errors:
            click.echo("Manifest validation errors:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
            raise SystemExit(1)
    except Exception as e:
        click.echo(f"Error loading manifest: {e}", err=True)
        raise SystemExit(1)

    click.echo(f"Source run: {reader.run_id}")
    click.echo(f"Source seed: {reader.run_seed}")
    click.echo(f"Stages: {', '.join(s.stage_name for s in reader.stages)}")

    # Determine replay mode
    mode_map = {
        "cache": ReplayMode.FULL_CACHE,
        "verify": ReplayMode.VERIFY,
        "from-stage": ReplayMode.FROM_STAGE,
        "selective": ReplayMode.SELECTIVE,
    }
    replay_mode = mode_map[mode]

    # Parse selective stages
    stages_list = stages.split(",") if stages else None

    # Create output directory
    if output is None:
        output = str(Path(reader.output_dir).parent / f"replay_{reader.run_id[:4]}")

    # Create replay plan
    import uuid

    new_run_id = str(uuid.uuid4())[:8]

    try:
        plan = create_replay_plan(
            manifest_path=manifest_file,
            new_run_id=new_run_id,
            output_dir=output,
            mode=replay_mode,
            from_stage=from_stage,
            stages_to_execute=stages_list,
        )
    except ValueError as e:
        click.echo(f"Error creating replay plan: {e}", err=True)
        raise SystemExit(1)

    click.echo(f"\nReplay plan created:")
    click.echo(f"  Mode: {replay_mode.value}")
    click.echo(f"  New run ID: {new_run_id}")
    click.echo(f"  Output: {output}")
    click.echo(f"  Cached stages: {len(plan.cached_stages)}")
    click.echo(f"  Execute stages: {len(plan.execute_stages)}")

    if plan.execute_stages:
        click.echo(f"  Will execute: {', '.join(sorted(plan.execute_stages))}")

    click.echo("\nReplay plan ready. Execute using the Python API.")


@main.command()
@click.argument("run_path")
@click.option("--stages/--no-stages", default=True, help="Show stage details")
@click.option("--cache/--no-cache", default=True, help="Show cache stats")
def inspect(run_path: str, stages: bool, cache: bool) -> None:
    """Inspect a run's manifests and outputs."""
    from pathlib import Path

    from labelforge.core.manifest.run_manifest import RunManifest

    run_dir = Path(run_path)
    manifest_path = run_dir / "manifest.json"

    if not manifest_path.exists():
        click.echo(f"Error: Manifest not found at {manifest_path}", err=True)
        raise SystemExit(1)

    try:
        manifest = RunManifest.load(manifest_path)
    except Exception as e:
        click.echo(f"Error loading manifest: {e}", err=True)
        raise SystemExit(1)

    click.echo("=== Run Manifest ===")
    click.echo(f"Run ID: {manifest.metadata.run_id}")
    click.echo(f"Run Name: {manifest.metadata.run_name or '(unnamed)'}")
    click.echo(f"Started: {manifest.metadata.started_at}")
    click.echo(f"Seed: {manifest.metadata.run_seed}")
    click.echo(f"Status: {manifest.status}")
    click.echo(f"Output Dir: {manifest.output_dir}")

    if manifest.metadata.git_commit:
        click.echo(f"\nGit Commit: {manifest.metadata.git_commit}")
        if manifest.metadata.git_branch:
            click.echo(f"Git Branch: {manifest.metadata.git_branch}")

    if stages and manifest.stages:
        click.echo(f"\n=== Stages ({len(manifest.stages)}) ===")
        for i, stage in enumerate(manifest.stages, 1):
            deps = f" (depends on: {', '.join(stage.depends_on)})" if stage.depends_on else ""
            click.echo(f"  {i}. {stage.stage_name} [{stage.stage_type}]{deps}")
            click.echo(f"     Version: {stage.stage_version}, Hash: {stage.stage_hash[:8]}...")


@main.command("diff")
@click.argument("run_a")
@click.argument("run_b")
@click.option("--output", "-o", help="Output report path")
def diff_runs(run_a: str, run_b: str, output: str | None) -> None:
    """Compare two runs and report differences."""
    from pathlib import Path

    from labelforge.core.manifest.run_manifest import RunManifest

    click.echo(f"Comparing runs: {run_a} vs {run_b}")

    # Load manifests
    manifest_a_path = Path(run_a) / "manifest.json"
    manifest_b_path = Path(run_b) / "manifest.json"

    if not manifest_a_path.exists():
        click.echo(f"Error: Manifest not found: {manifest_a_path}", err=True)
        raise SystemExit(1)
    if not manifest_b_path.exists():
        click.echo(f"Error: Manifest not found: {manifest_b_path}", err=True)
        raise SystemExit(1)

    manifest_a = RunManifest.load(manifest_a_path)
    manifest_b = RunManifest.load(manifest_b_path)

    click.echo(f"\nRun A: {manifest_a.metadata.run_id} (seed: {manifest_a.metadata.run_seed})")
    click.echo(f"Run B: {manifest_b.metadata.run_id} (seed: {manifest_b.metadata.run_seed})")

    # Compare seeds
    if manifest_a.metadata.run_seed != manifest_b.metadata.run_seed:
        click.echo("\n⚠️  Seeds differ!")

    # Compare stages
    stages_a = {s.stage_name: s for s in manifest_a.stages}
    stages_b = {s.stage_name: s for s in manifest_b.stages}

    added = set(stages_b.keys()) - set(stages_a.keys())
    removed = set(stages_a.keys()) - set(stages_b.keys())
    common = set(stages_a.keys()) & set(stages_b.keys())

    if added:
        click.echo(f"\n+ Added stages: {', '.join(added)}")
    if removed:
        click.echo(f"\n- Removed stages: {', '.join(removed)}")

    # Compare common stages
    changed = []
    for name in common:
        if stages_a[name].stage_hash != stages_b[name].stage_hash:
            changed.append(name)

    if changed:
        click.echo(f"\n~ Modified stages: {', '.join(changed)}")
    elif not added and not removed:
        click.echo("\n✓ Stage configurations match")


@main.group()
def prompt() -> None:
    """Prompt pack management commands."""
    pass


@prompt.command("lint")
@click.argument("pack_path")
def prompt_lint(pack_path: str) -> None:
    """Validate a prompt pack."""
    from pathlib import Path

    from labelforge.core.prompt_resolver import PromptResolver

    pack_dir = Path(pack_path).parent
    pack_name = Path(pack_path).parent.name

    try:
        resolver = PromptResolver(pack_dir.parent)
        errors = resolver.validate_pack(pack_name)

        if errors:
            click.echo("Validation errors:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
            raise SystemExit(1)
        else:
            click.echo(f"Prompt pack '{pack_name}' is valid")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)


@prompt.command("list")
@click.option("--prompts-dir", "-d", default="prompts", help="Prompts directory")
def prompt_list(prompts_dir: str) -> None:
    """List available prompt packs."""
    from pathlib import Path

    from labelforge.core.prompt_resolver import PromptResolver

    resolver = PromptResolver(Path(prompts_dir))
    packs = resolver.list_packs()

    if packs:
        click.echo("Available prompt packs:")
        for pack in packs:
            click.echo(f"  - {pack}")
    else:
        click.echo("No prompt packs found")


@main.group()
def cache() -> None:
    """Cache management commands."""
    pass


@cache.command("stats")
@click.option("--cache-dir", "-d", default=".cache", help="Cache directory")
def cache_stats(cache_dir: str) -> None:
    """Show cache statistics."""
    from pathlib import Path

    from labelforge.cache.fs_cache import FilesystemCache

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        click.echo("Cache directory does not exist")
        return

    try:
        fs_cache = FilesystemCache(cache_path)
        stats = fs_cache.get_stats()

        click.echo(f"Cache directory: {stats['cache_dir']}")
        click.echo(f"Total entries: {stats['total_entries']}")
        click.echo(f"Total size: {stats['total_size_bytes'] / 1024 / 1024:.2f} MB")

        if stats.get("by_stage"):
            click.echo("\nBy stage:")
            for stage, stage_stats in stats["by_stage"].items():
                size_mb = stage_stats["size"] / 1024 / 1024
                click.echo(f"  {stage}: {stage_stats['count']} entries ({size_mb:.2f} MB)")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


@cache.command("clear")
@click.option("--cache-dir", "-d", default=".cache", help="Cache directory")
@click.option("--stage", "-s", help="Only clear specific stage")
@click.confirmation_option(prompt="Are you sure you want to clear the cache?")
def cache_clear(cache_dir: str, stage: str | None) -> None:
    """Clear cache entries."""
    from pathlib import Path

    from labelforge.cache.fs_cache import FilesystemCache

    cache_path = Path(cache_dir)
    if not cache_path.exists():
        click.echo("Cache directory does not exist")
        return

    try:
        fs_cache = FilesystemCache(cache_path)

        if stage:
            count = fs_cache.clear_stage(stage)
            click.echo(f"Cleared {count} entries for stage '{stage}'")
        else:
            # Clear all
            import shutil

            shutil.rmtree(cache_path)
            click.echo("Cache cleared")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)


if __name__ == "__main__":
    main()
