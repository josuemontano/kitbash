"""Interactive user prompts routed through the live dashboard."""

from collections.abc import Sequence
from pathlib import Path

from rich.console import Console, RenderableType
from rich.rule import Rule
from rich.text import Text

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
        content: list[RenderableType] = [
            Rule(f"[bold]Review {asset.id}[/bold] — {item.name}  ({review.queue_length} more waiting)"),
            f"{item.description}\n[dim]{review.loop_message}[/dim]",
        ]
        if asset.error:
            content.append(f"[red]Last error:[/red] {asset.error}")
        if review.scorecard:
            content.append(scorecard_table(review.scorecard))
            facts = review.scorecard.facts
            if "usd_roundtrip_score" in facts:
                content.append(
                    f"USD: {facts.get('usd_material_mode', '?')}, round trip {facts['usd_roundtrip_score']:.2f}, "
                    f"size {facts.get('dimensions_m', '?')} m"
                )
        previews = [*review.previews[:2], *([review.usd_compare] if review.usd_compare else [])]
        if not review.committable:
            content.append("[yellow]There is no complete build (.blend, .usd, preview) to approve yet.[/yellow]")
        choices = ["a", "f", "r", "s", "v"] if review.committable else ["f", "r", "s", "v"]
        while True:
            answer = self._ask(
                "[a]pprove, [f]eedback, [r]egenerate, [s]kip, [v]iew all images",
                choices=choices, default=choices[0], content=content, images=previews,
            )
            match answer:
                case "a":
                    return ReviewDecision(ReviewAction.APPROVE)
                case "f":
                    return ReviewDecision(ReviewAction.FEEDBACK, feedback=self._text("What should change?"))
                case "r":
                    note = self._ask("Anything the new version must do differently? (optional)", default="")
                    return ReviewDecision(ReviewAction.REGENERATE, feedback=note)
                case "s":
                    return ReviewDecision(ReviewAction.SKIP)
                case "v":
                    paths = [*review.previews, *([review.reference] if review.reference else [])]
                    if not self._dashboard.enabled:
                        for path in paths:
                            self._console.print(str(path))
                    self.show_images(paths)
            if not self._dashboard.enabled:
                content = []
                previews = []

    def provide_input(self, asset: AssetRecord, item: InventoryItem, request: str) -> ReviewDecision:
        content: list[RenderableType] = [Rule(f"[bold magenta]Input needed: {asset.id}[/bold magenta]"), request]
        review = asset.extra.get("reference_review") or {}
        candidates = review.get("candidates", [])
        images = [Path(review["contact_sheet"])] if review.get("contact_sheet") else []
        if candidates:
            rows = tuple(
                (
                    str(index + 1), candidate.get("title", ""),
                    f"{candidate.get('license_id') or candidate.get('license') or 'unknown'} "
                    f"({'allowed' if candidate.get('rights_allowed') else 'not allowed'})",
                    candidate.get("creator") or "unknown", f"{candidate.get('quality', {}).get('score', 0):.2f}",
                )
                for index, candidate in enumerate(candidates)
            )
            content.append(simple_table("Reference candidates (heuristic scores)", ("#", "title", "rights", "creator", "score"), rows))
        choices = [str(index + 1) for index in range(len(candidates))] + ["n", "p", "m", "s"]
        label = "Candidate number, " if candidates else ""
        answer = self._ask(
            label + "n: name to search for, p: path to a reference image (preferred), "
            "m: model it programmatically instead of with Trellis, s: skip",
            choices=choices, default="n", content=content, images=images,
        )
        if answer.isdecimal():
            return ReviewDecision(ReviewAction.PROVIDE_INPUT, reference_index=int(answer) - 1)
        match answer:
            case "n":
                return ReviewDecision(ReviewAction.PROVIDE_INPUT, search_name=self._text("Item name to search for"))
            case "p":
                return ReviewDecision(ReviewAction.PROVIDE_INPUT, reference_path=str(self._existing_file("Reference image path")))
            case "m":
                return ReviewDecision(ReviewAction.PROVIDE_INPUT, procedural=True)
        return ReviewDecision(ReviewAction.SKIP)

    # -- phase gates -------------------------------------------------------------------------------

    def confirm(self, summary: PhaseSummary) -> GateDecision:
        content: list[RenderableType] = [Rule(f"[bold]{summary.phase.value.capitalize()} gate[/bold]"), summary.headline]
        if summary.message:
            content.append(f"[dim]{summary.message}[/dim]")
        if summary.rows:
            content.append(simple_table(summary.phase.value, summary.columns, summary.rows))
        if summary.scorecard:
            content.append(scorecard_table(summary.scorecard))
        if summary.phase is PhaseName.ASSEMBLY:
            answer = self._ask(
                "p: publish degraded scene despite failed validation; q: quit without accepting",
                choices=["p", "q"], default="q", content=content, images=summary.images,
            )
            return GateDecision(GateAction.PUBLISH_DEGRADED if answer == "p" else GateAction.ABORT)
        choices = ["a", "f", "q"] if summary.phase is not PhaseName.MODELLING else ["a", "r", "q"]
        labels = "[a]pprove and continue, [f]eedback, [q]uit" if "f" in choices else "[a]pprove and continue, [r]ework an asset, [q]uit"
        answer = self._ask(labels, choices=choices, default="a", content=content, images=summary.images)
        match answer:
            case "a":
                return GateDecision(GateAction.APPROVE)
            case "f":
                return GateDecision(GateAction.FEEDBACK, feedback=self._text("What should change?"))
            case "r":
                asset_id = self._ask("Asset id", choices=list(summary.asset_ids))
                return GateDecision(GateAction.REWORK_ASSET, feedback=self._text("What should change?"), asset_id=asset_id)
        return GateDecision(GateAction.ABORT)

    # -- inventory questions -------------------------------------------------------------------------

    def confirm_reuse(self, item: InventoryItem, hit: SearchHit) -> bool:
        content: list[RenderableType] = [
            f"[bold]{item.id}[/bold] ({item.name}) looks like backlot asset [bold]{hit.entry.id}[/bold] "
            f"({hit.entry.name}, similarity {hit.similarity:.2f})."
        ]
        images = [hit.entry.preview_path]
        while True:
            answer = self._ask(
                "Reuse it instead of modelling a new one? [y/n]", default="y",
                content=content, images=images, columns=30,
            ).strip().lower()
            if answer in ("y", "yes", "n", "no"):
                return answer in ("y", "yes")
            content = [Text("Please enter Y or N", style="red")]
            images = []

    def resolve_unrecognized(self, item: InventoryItem) -> UnrecognizedAnswer:
        answer = self._ask(
            "[k]eep as is, give a [n]ame to search for, a reference image [p]ath, or [d]rop it",
            choices=["k", "n", "p", "d"], default="n",
            content=[
                f"[bold magenta]Unrecognized item[/bold magenta] {item.id}: '{item.name}' "
                f"(confidence {item.confidence:.2f}). {item.description}"
            ],
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
        if self._dashboard.enabled:
            self._dashboard.open_images(images)
        else:
            self._images.show(images)

    def _ask(
        self,
        question: str,
        *,
        choices: Sequence[str] | None = None,
        default: str | None = None,
        content: Sequence[RenderableType] = (),
        images: Sequence[Path] = (),
        columns: int = 60,
    ) -> str:
        if not self._dashboard.enabled:
            self._images.show(images, columns=columns)
        return self._dashboard.ask(question, choices=choices, default=default, content=content, images=images)

    def _text(self, question: str) -> str:
        content: Sequence[RenderableType] = ()
        while not (answer := self._ask(question, content=content).strip()):
            content = [Text("Please type something.", style="red")]
        return answer

    def _existing_file(self, question: str) -> Path:
        content: Sequence[RenderableType] = ()
        while True:
            path = Path(self._ask(question, content=content).strip()).expanduser()
            if path.is_file():
                return path.resolve()
            content = [Text(f"No file at {path}", style="red")]
