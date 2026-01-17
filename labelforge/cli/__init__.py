"""
LabelForge CLI.

Command-line interface for running pipelines, replays, and inspections.
"""

import click

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
def run(
    config: str,
    output: str,
    seed: int,
    name: str | None,
    deterministic: bool,
    cache: bool,
) -> None:
    """Run a labeling pipeline."""
    click.echo(f"Running pipeline from config: {config}")
    click.echo(f"Output directory: {output}")
    click.echo(f"Seed: {seed}")
    click.echo(f"Deterministic: {deterministic}")
    click.echo(f"Cache enabled: {cache}")

    # TODO: Implement pipeline execution
    click.echo("Pipeline execution not yet implemented")


@main.command()
@click.argument("manifest_path")
@click.option("--output", "-o", help="Output directory (defaults to original)")
@click.option("--validate/--no-validate", default=True, help="Validate environment match")
def replay(manifest_path: str, output: str | None, validate: bool) -> None:
    """Replay a previous run from its manifest."""
    click.echo(f"Replaying from manifest: {manifest_path}")

    # TODO: Implement replay
    click.echo("Replay not yet implemented")


@main.command()
@click.argument("run_path")
@click.option("--stages/--no-stages", default=True, help="Show stage details")
@click.option("--cache/--no-cache", default=True, help="Show cache stats")
def inspect(run_path: str, stages: bool, cache: bool) -> None:
    """Inspect a run's manifests and outputs."""
    click.echo(f"Inspecting run: {run_path}")

    # TODO: Implement inspection
    click.echo("Inspection not yet implemented")


@main.command("diff")
@click.argument("run_a")
@click.argument("run_b")
@click.option("--output", "-o", help="Output report path")
def diff_runs(run_a: str, run_b: str, output: str | None) -> None:
    """Compare two runs and report differences."""
    click.echo(f"Comparing runs: {run_a} vs {run_b}")

    # TODO: Implement diff
    click.echo("Diff not yet implemented")


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
