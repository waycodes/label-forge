"""
Ablation CLI command.

Run ablation experiments with config matrix expansion and shared caching.
"""

import click
import yaml
from pathlib import Path
from datetime import datetime

from labelforge.experiments.matrix import (
    AblationMatrix,
    AblationRun,
    parse_override_string,
)
from labelforge.core.json_canonical import canonical_json_dumps


@click.group("ablate")
def ablate() -> None:
    """Run ablation experiments with config variations."""
    pass


@ablate.command("run")
@click.argument("matrix_config", type=click.Path(exists=True))
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="ablations",
    help="Output directory for ablation results",
)
@click.option(
    "--cache-dir",
    type=click.Path(),
    default=".cache",
    help="Shared cache directory across variants",
)
@click.option(
    "--parallel", "-p",
    type=int,
    default=1,
    help="Number of variants to run in parallel",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be run without executing",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume a previous ablation run",
)
def run_ablation(
    matrix_config: str,
    output: str,
    cache_dir: str,
    parallel: int,
    dry_run: bool,
    resume: bool,
) -> None:
    """
    Run an ablation study from a matrix configuration.

    MATRIX_CONFIG is a YAML file defining the ablation matrix:

    \b
    name: temperature_sweep
    description: Test different sampling temperatures
    base_config:
      model: meta-llama/Llama-2-7b-chat-hf
      seed: 42
    overrides:
      temperature: [0.0, 0.5, 1.0]
      top_p: [0.9, 0.95]
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Load matrix
    matrix = AblationMatrix.from_yaml(matrix_config)
    variants = matrix.expand()

    console.print(f"\n[bold]Ablation Study: {matrix.name}[/bold]")
    if matrix.description:
        console.print(f"[dim]{matrix.description}[/dim]")
    console.print(f"\nTotal variants: [cyan]{len(variants)}[/cyan]")

    # Show variant table
    table = Table(title="Experiment Variants")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Overrides")

    for variant in variants:
        override_str = ", ".join(f"{k}={v}" for k, v in variant.overrides.items())
        table.add_row(variant.variant_id, variant.variant_name, override_str)

    console.print(table)

    if dry_run:
        console.print("\n[yellow]Dry run - no experiments executed[/yellow]")
        return

    # Create output directory
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create ablation run
    ablation_run = AblationRun(matrix=matrix, variants=variants)

    # Check for resume
    run_state_path = output_dir / "ablation_state.json"
    if resume and run_state_path.exists():
        console.print("\n[yellow]Resuming from previous state...[/yellow]")
        # Load previous state (simplified - in production would deserialize properly)

    # Save initial state
    save_ablation_state(ablation_run, run_state_path)

    # Run variants
    console.print(f"\n[bold]Running {len(ablation_run.pending_variants)} variants...[/bold]\n")

    for i, variant in enumerate(ablation_run.pending_variants):
        console.print(f"[{i+1}/{len(variants)}] Running variant: [cyan]{variant.variant_name}[/cyan]")

        try:
            # Create variant config
            variant_config = variant.apply_to_config(matrix.base_config)

            # Create variant output directory
            variant_dir = output_dir / variant.variant_id
            variant_dir.mkdir(parents=True, exist_ok=True)

            # Save variant config
            config_path = variant_dir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(variant_config, f, default_flow_style=False)

            # Run pipeline (placeholder - would integrate with runner)
            run_id = run_variant_pipeline(
                config=variant_config,
                output_dir=variant_dir,
                cache_dir=cache_dir,
                console=console,
            )

            ablation_run.mark_complete(variant.variant_id, run_id)
            console.print(f"  [green]✓ Complete[/green] (run_id: {run_id})")

        except Exception as e:
            ablation_run.mark_failed(variant.variant_id, str(e))
            console.print(f"  [red]✗ Failed: {e}[/red]")

        # Save state after each variant
        save_ablation_state(ablation_run, run_state_path)

    # Final summary
    ablation_run.completed_at = datetime.utcnow()
    save_ablation_state(ablation_run, run_state_path)

    console.print(f"\n[bold]Ablation Complete[/bold]")
    console.print(f"  Completed: [green]{ablation_run.completed_count}[/green]")
    console.print(f"  Failed: [red]{ablation_run.failed_count}[/red]")
    console.print(f"  Results in: [cyan]{output_dir}[/cyan]")


@ablate.command("expand")
@click.argument("matrix_config", type=click.Path(exists=True))
@click.option("--format", "-f", type=click.Choice(["table", "json", "yaml"]), default="table")
def expand_matrix(matrix_config: str, format: str) -> None:
    """
    Expand a matrix config and show all variants.

    Useful for previewing what experiments will be run.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    matrix = AblationMatrix.from_yaml(matrix_config)
    variants = matrix.expand()

    if format == "table":
        table = Table(title=f"Variants for {matrix.name}")
        table.add_column("ID")
        table.add_column("Name")
        table.add_column("Overrides")
        table.add_column("Hash")

        for variant in variants:
            override_str = ", ".join(f"{k}={v}" for k, v in variant.overrides.items())
            table.add_row(
                variant.variant_id,
                variant.variant_name,
                override_str,
                variant.unique_id,
            )

        console.print(table)

    elif format == "json":
        data = [v.to_dict() for v in variants]
        console.print(canonical_json_dumps(data, indent=True))

    elif format == "yaml":
        data = [v.to_dict() for v in variants]
        console.print(yaml.dump(data, default_flow_style=False))


@ablate.command("status")
@click.argument("ablation_dir", type=click.Path(exists=True))
def ablation_status(ablation_dir: str) -> None:
    """Show status of an ablation run."""
    from rich.console import Console
    from rich.table import Table

    console = Console()

    ablation_path = Path(ablation_dir)
    state_path = ablation_path / "ablation_state.json"

    if not state_path.exists():
        console.print("[red]No ablation state found[/red]")
        return

    from labelforge.core.json_canonical import canonical_json_loads

    state = canonical_json_loads(state_path.read_text())

    console.print(f"\n[bold]Ablation: {state['matrix']['name']}[/bold]")
    console.print(f"Started: {state['started_at']}")
    if state.get("completed_at"):
        console.print(f"Completed: {state['completed_at']}")

    table = Table(title="Variant Status")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Run ID / Error")

    completed = state.get("completed", {})
    failed = state.get("failed", {})

    for variant in state.get("variants", []):
        vid = variant["variant_id"]
        name = variant["variant_name"]

        if vid in completed:
            status = "[green]Complete[/green]"
            info = completed[vid]
        elif vid in failed:
            status = "[red]Failed[/red]"
            info = failed[vid][:50] + "..." if len(failed[vid]) > 50 else failed[vid]
        else:
            status = "[yellow]Pending[/yellow]"
            info = "-"

        table.add_row(vid, name, status, info)

    console.print(table)


def run_variant_pipeline(
    config: dict,
    output_dir: Path,
    cache_dir: str,
    console,
) -> str:
    """
    Run a single variant pipeline.

    This is a placeholder that would integrate with the full pipeline runner.
    """
    import uuid

    # Generate run ID
    run_id = str(uuid.uuid4())[:8]

    # In a full implementation, this would:
    # 1. Load pipeline config
    # 2. Create PipelineRunner
    # 3. Execute with shared cache
    # 4. Return run_id

    # For now, just create a marker file
    marker = output_dir / "run_id.txt"
    marker.write_text(run_id)

    return run_id


def save_ablation_state(run: AblationRun, path: Path) -> None:
    """Save ablation run state to file."""
    state = run.to_dict()
    path.write_text(canonical_json_dumps(state, indent=True))
