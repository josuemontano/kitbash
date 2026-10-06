"""Command line interface."""

import re
from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from kitbash.app import Application, create_workspace, open_workspace
from kitbash.backlot.library import Backlot
from kitbash.config import default_rubric_path, load_config, parse_model_overrides
from kitbash.domain.phases import PhaseName
from kitbash.domain.rubric import Rubric
from kitbash.domain.run_input import RunInput
from kitbash.errors import ConfigError, KitbashError
from kitbash.infra.embeddings import make_embedder
from kitbash.paths import OutputLayout
from kitbash.services.plan import Planner
from kitbash.ui.summary import print_plan, print_summary
from kitbash.ui.tables import backlot_table, search_table, simple_table

console = Console()
app = typer.Typer(no_args_is_help=True, add_completion=False, help="Turn an image or a prompt into an editable Blender scene.")
library_app = typer.Typer(no_args_is_help=True, help="The backlot: the reusable asset library.")
app.add_typer(library_app, name="library")
retopology_app = typer.Typer(no_args_is_help=True, help="Retopology tools.")
app.add_typer(retopology_app, name="retopology")

MODEL_FLAG = re.compile(r"^--model\.([\w.-]+?)(?:=(.*))?$")


class Style(StrEnum):
    photorealistic = "photorealistic"
    two_d = "2d"
    animated_3d = "animated-3d"


class Retopology(StrEnum):
    triflow = "triflow"
    decimate = "decimate"


def _guard(action: Callable[[], Any]) -> Any:
    """Run a command with clear error messages, and stop child processes on Ctrl+C."""
    try:
        return action()
    except KitbashError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Progress is checkpointed; continue with `kitbash resume --output <dir>`.[/yellow]")
        raise typer.Exit(130) from None


def parse_extra_args(args: Sequence[str]) -> dict[str, str]:
    """``--model.<role>=<name>`` and ``--model.<phase>.<role>=<name>`` (also with a space instead of '=')."""
    pairs: dict[str, str] = {}
    items = list(args)
    while items:
        arg = items.pop(0)
        match = MODEL_FLAG.match(arg)
        if not match:
            raise ConfigError(f"Unknown option {arg!r}", hint="See `kitbash build --help`.")
        key, value = match.group(1), match.group(2)
        if value is None:
            if not items:
                raise ConfigError(f"{arg} needs a model name")
            value = items.pop(0)
        pairs[key] = value
    return parse_model_overrides(pairs)


def _overrides(ctx: typer.Context, **options: Any) -> dict[str, Any]:
    keys = {
        "style": "pipeline.style",
        "threads": "pipeline.threads",
        "review_buffer": "pipeline.review_buffer",
        "max_cycles": "critic.max_cycles",
        "retopology": "retopology.method",
    }
    overrides = {keys[name]: (value.value if isinstance(value, StrEnum) else value) for name, value in options.items() if value is not None}
    return overrides | parse_extra_args(ctx.args)


def _run(application: Application, from_phase: PhaseName | None = None) -> None:
    try:
        report = application.run(from_phase)
    finally:
        application.close()
    print_summary(console, report)
    layout = application.layout
    acceptance = report.get("scene", {}).get("acceptance", {})
    label = "[bold yellow]Published degraded scene (human override).[/bold yellow]" if acceptance.get("status") == "overridden" else "[bold green]Done.[/bold green]"
    console.print(
        f"\n{label} Scene: {layout.scene_blend}\nUSD: {layout.scene_usd}\n"
        f"Analytics: {layout.analytics_md}"
    )


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def build(
    ctx: typer.Context,
    output: Annotated[Path, typer.Option("--output", help="Scene output directory.")],
    image: Annotated[Path | None, typer.Option("--image", help="Reference image.")] = None,
    prompt: Annotated[str | None, typer.Option("--prompt", help="Text prompt (no reference image).")] = None,
    style: Annotated[Style | None, typer.Option("--style", help="Render style (default: photorealistic).")] = None,
    threads: Annotated[int | None, typer.Option("--threads", help="Parallel modelling workers / Trellis jobs (default 2).")] = None,
    review_buffer: Annotated[int | None, typer.Option("--review-buffer", help="Max assets waiting for review (default 6).")] = None,
    max_cycles: Annotated[int | None, typer.Option("--max-cycles", help="Critic cycles per phase and per asset (default 4).")] = None,
    retopology: Annotated[
        Retopology | None, typer.Option("--retopology", help="Retopology of the Trellis mesh (default: triflow; decimate = in Blender only).")
    ] = None,
    rubric: Annotated[Path | None, typer.Option("--rubric", help="Rubric Markdown file.")] = None,
    config: Annotated[Path | None, typer.Option("--config", help="Config TOML overriding the defaults.")] = None,
    no_interactive: Annotated[bool, typer.Option("--no-interactive", help="Auto-approve every gate and review.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the plan and exit without running anything.")] = False,
) -> None:
    """Build a scene from --image or --prompt. Extra options: --model.<role>=<name>, --model.<phase>.<role>=<name>."""

    def action() -> None:
        run_input = RunInput.create(image, prompt)
        overrides = _overrides(
            ctx, style=style, threads=threads, review_buffer=review_buffer, max_cycles=max_cycles, retopology=retopology
        )
        if dry_run:
            settings = load_config(config, overrides)
            planner = Planner(settings, OutputLayout.at(output), run_input, Rubric.load(rubric or default_rubric_path()))
            print_plan(console, planner)
            return
        settings, layout, local_input = create_workspace(output, run_input, config, rubric, overrides)
        _run(Application(settings, layout, local_input, interactive=not no_interactive, console=console))

    _guard(action)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def resume(
    ctx: typer.Context,
    output: Annotated[Path, typer.Option("--output", help="Scene output directory of an earlier run.")],
    from_phase: Annotated[PhaseName | None, typer.Option("--from-phase", help="Re-open this phase and every later one.")] = None,
    no_interactive: Annotated[bool, typer.Option("--no-interactive", help="Auto-approve every gate and review.")] = False,
) -> None:
    """Continue a run from its checkpoints (assets restart from their last completed step)."""

    def action() -> None:
        settings, layout, run_input = open_workspace(output, parse_extra_args(ctx.args))
        _run(Application(settings, layout, run_input, interactive=not no_interactive, console=console), from_phase)

    _guard(action)


# -- library ------------------------------------------------------------------------------------------------

ConfigOption = Annotated[Path | None, typer.Option("--config", help="Config TOML (backlot path, embedding backend).")]


def _backlot(config: Path | None, *, rebuild_index: bool = False) -> Backlot:
    settings = load_config(config)
    return Backlot(settings.paths.backlot, make_embedder(settings.embedding), rebuild_index=rebuild_index)


@library_app.command("search")
def library_search(
    query: str,
    k: Annotated[int, typer.Option("--k", help="Number of results.")] = 5,
    config: ConfigOption = None,
) -> None:
    """Semantic search of the asset library."""
    _guard(lambda: console.print(search_table(_backlot(config).search(query, k=k))))


@library_app.command("list")
def library_list(config: ConfigOption = None) -> None:
    """List every asset in the library."""
    _guard(lambda: console.print(backlot_table(_backlot(config).list())))


@library_app.command("show")
def library_show(asset_id: str, config: ConfigOption = None) -> None:
    """Show one asset's metadata and files."""

    def action() -> None:
        entry = _backlot(config).get(asset_id)
        rows = [(k, v) for k, v in entry.to_dict().items() if k != "metadata"]
        console.print(simple_table(entry.name, ("field", "value"), rows))

    _guard(action)


@library_app.command("remove")
def library_remove(
    asset_id: str, yes: Annotated[bool, typer.Option("--yes", help="Do not ask for confirmation.")] = False, config: ConfigOption = None
) -> None:
    """Delete an asset and its files from the library."""

    def action() -> None:
        backlot = _backlot(config)
        entry = backlot.get(asset_id)
        if not yes and not typer.confirm(f"Delete {entry.id} ({entry.name}) and its files?"):
            raise typer.Abort()
        backlot.remove(asset_id)
        console.print(f"Removed {asset_id}")

    _guard(action)


@library_app.command("reindex")
def library_reindex(config: ConfigOption = None) -> None:
    """Re-embed every asset with the current embedding backend."""
    _guard(lambda: console.print(f"Re-indexed {_backlot(config, rebuild_index=True).reindex()} assets"))


# -- retopology ---------------------------------------------------------------------------------------------


@retopology_app.command("download-weights")
def retopology_download_weights(config: ConfigOption = None) -> None:
    """Download and verify the TriFlow weights (about 1.3 GB, once). Otherwise they are fetched on first use."""

    def action() -> None:
        from kitbash.retopology.triflow import weights

        directory = load_config(config).paths.triflow_weights
        console.print(f"Fetching TriFlow weights into {directory} ...")
        paths = weights.ensure(directory)
        for name, path in paths.items():
            console.print(f"  {name}: {path} ({path.stat().st_size / 1e6:,.0f} MB, sha256 verified)")

    _guard(action)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
