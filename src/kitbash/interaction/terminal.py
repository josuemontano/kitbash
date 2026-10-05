"""Interactive terminal user. Every prompt pauses the live display first."""

from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt

from kitbash.backlot.library import SearchHit
from kitbash.domain.assets import AssetRecord
from kitbash.domain.inventory import InventoryItem
from kitbash.domain.phases import PhaseName
from kitbash.interaction.protocols import (
    AssetReview,
    GateAction,
    GateDecision,
    PhaseSummary,
    ReviewAction,
    ReviewDecision,
    UnrecognizedAction,
    UnrecognizedAnswer,
)
from kitbash.ui.dashboard import Dashboard
from kitbash.ui.images import ImagePresenter
from kitbash.ui.tables import scorecard_table, simple_table


class TerminalUser:
    interactive = True

    def __init__(self, console: Console, dashboard: Dashboard, images: ImagePresenter) -> None:
        self._console = console
        self._dashboard = dashboard
        self._images = images

    # -- asset reviews -----------------------------------------------------------------------------

    def review(self, review: AssetReview) -> ReviewDecision:
        asset, item = review.asset, review.item
        with self._dashboard.paused():
            console = self._console
            console.rule(f"[bold]Review {asset.id}[/bold] — {item.name}  ({review.queue_length} more waiting)")
            console.print(f"{item.description}\n[dim]{review.loop_message}[/dim]")
            if asset.error:
                console.print(f"[red]Last error:[/red] {asset.error}")
            if review.scorecard:
                console.print(scorecard_table(review.scorecard))
                facts = review.scorecard.facts
                if "usd_roundtrip_score" in facts:
                    console.print(
                        f"USD: {facts.get('usd_material_mode', '?')}, round trip {facts['usd_roundtrip_score']:.2f}, "
                        f"size {facts.get('dimensions_m', '?')} m"
                    )
            self._images.show([*review.previews[:2], *([review.usd_compare] if review.usd_compare else [])])
            if not review.committable:
                console.print("[yellow]There is no complete build (.blend, .usd, preview) to approve yet.[/yellow]")
            choices = ["a", "f", "r", "s", "v"] if review.committable else ["f", "r", "s", "v"]
            while True:
                answer = Prompt.ask(
                    "[a]pprove, [f]eedback, [r]egenerate, [s]kip, [v]iew all images",
                    choices=choices,
                    default=choices[0],
                    console=console,
                )
                match answer:
                    case "a":
                        return ReviewDecision(ReviewAction.APPROVE)
                    case "f":
                        return ReviewDecision(ReviewAction.FEEDBACK, feedback=self._text("What should change?"))
                    case "r":
                        note = Prompt.ask("Anything the new version must do differently? (optional)", default="", console=console)
                        return ReviewDecision(ReviewAction.REGENERATE, feedback=note)
                    case "s":
                        return ReviewDecision(ReviewAction.SKIP)
                    case "v":
                        paths = [*review.previews, *([review.reference] if review.reference else [])]
                        for path in paths:
                            console.print(str(path))
                        self._images.show(paths)

    def provide_input(self, asset: AssetRecord, item: InventoryItem, request: str) -> ReviewDecision:
        with self._dashboard.paused():
            self._console.rule(f"[bold magenta]Input needed: {asset.id}[/bold magenta]")
            self._console.print(request)
            answer = Prompt.ask("[n]ame to search for, [p]ath to a reference image, [s]kip", choices=["n", "p", "s"], default="n", console=self._console)
            match answer:
                case "n":
                    return ReviewDecision(ReviewAction.PROVIDE_INPUT, search_name=self._text("Item name to search for"))
                case "p":
                    return ReviewDecision(ReviewAction.PROVIDE_INPUT, reference_path=str(self._existing_file("Reference image path")))
            return ReviewDecision(ReviewAction.SKIP)

    # -- phase gates -------------------------------------------------------------------------------

    def confirm(self, summary: PhaseSummary) -> GateDecision:
        with self._dashboard.paused():
            console = self._console
            console.rule(f"[bold]{summary.phase.value.capitalize()} gate[/bold]")
            console.print(summary.headline)
            if summary.message:
                console.print(f"[dim]{summary.message}[/dim]")
            if summary.rows:
                console.print(simple_table(summary.phase.value, summary.columns, summary.rows))
            if summary.scorecard:
                console.print(scorecard_table(summary.scorecard))
            self._images.show(summary.images)
            choices = ["a", "f", "q"] if summary.phase is not PhaseName.MODELLING else ["a", "r", "q"]
            labels = "[a]pprove and continue, [f]eedback, [q]uit" if "f" in choices else "[a]pprove and continue, [r]ework an asset, [q]uit"
            answer = Prompt.ask(labels, choices=choices, default="a", console=console)
            match answer:
                case "a":
                    return GateDecision(GateAction.APPROVE)
                case "f":
                    return GateDecision(GateAction.FEEDBACK, feedback=self._text("What should change?"))
                case "r":
                    asset_id = Prompt.ask("Asset id", choices=list(summary.asset_ids), console=console)
                    return GateDecision(GateAction.REWORK_ASSET, feedback=self._text("What should change?"), asset_id=asset_id)
            return GateDecision(GateAction.ABORT)

    # -- inventory questions -------------------------------------------------------------------------

    def confirm_reuse(self, item: InventoryItem, hit: SearchHit) -> bool:
        with self._dashboard.paused():
            self._console.print(
                f"[bold]{item.id}[/bold] ({item.name}) looks like backlot asset [bold]{hit.entry.id}[/bold] "
                f"({hit.entry.name}, similarity {hit.similarity:.2f})."
            )
            self._images.show([hit.entry.preview_path], columns=30)
            return Confirm.ask("Reuse it instead of modelling a new one?", default=True, console=self._console)

    def resolve_unrecognized(self, item: InventoryItem) -> UnrecognizedAnswer:
        with self._dashboard.paused():
            self._console.print(
                f"[bold magenta]Unrecognized item[/bold magenta] {item.id}: '{item.name}' (confidence {item.confidence:.2f}). {item.description}"
            )
            answer = Prompt.ask(
                "[k]eep as is, give a [n]ame to search for, a reference image [p]ath, or [d]rop it",
                choices=["k", "n", "p", "d"],
                default="n",
                console=self._console,
            )
            match answer:
                case "n":
                    return UnrecognizedAnswer(UnrecognizedAction.SEARCH_NAME, self._text("What is it?"))
                case "p":
                    return UnrecognizedAnswer(UnrecognizedAction.REFERENCE, str(self._existing_file("Reference image path")))
                case "d":
                    return UnrecognizedAnswer(UnrecognizedAction.DROP)
            return UnrecognizedAnswer(UnrecognizedAction.KEEP)

    # -- misc ----------------------------------------------------------------------------------------

    def notify(self, message: str) -> None:
        self._dashboard.log(message)

    def show_images(self, images: Sequence[Path]) -> None:
        with self._dashboard.paused():
            self._images.show(images)

    def _text(self, question: str) -> str:
        while not (answer := Prompt.ask(question, console=self._console).strip()):
            self._console.print("[red]Please type something.[/red]")
        return answer

    def _existing_file(self, question: str) -> Path:
        while True:
            path = Path(Prompt.ask(question, console=self._console).strip()).expanduser()
            if path.is_file():
                return path.resolve()
            self._console.print(f"[red]No file at {path}[/red]")
