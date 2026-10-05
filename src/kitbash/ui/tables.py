"""Rich tables shared by the CLI, the gates and the final summary."""

from collections.abc import Iterable, Sequence
from typing import Any

from rich.table import Table
from rich.text import Text

from kitbash.backlot.library import BacklotEntry, SearchHit
from kitbash.domain.critique import ScoreCard
from kitbash.domain.inventory import Inventory


def simple_table(title: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> Table:
    table = Table(title=title, expand=False)
    for column in columns:
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(*(value if isinstance(value, Text) else str(value) for value in row))
    return table


def inventory_table(inventory: Inventory, threshold: float) -> Table:
    rows = []
    for item in inventory.items:
        confidence = Text(f"{item.confidence:.2f}", style="red" if item.confidence < threshold else "")
        size = "{:.2f} x {:.2f} x {:.2f}".format(*item.dimensions.as_tuple())
        note = []
        if item.same_as:
            note.append(f"copy of {item.same_as}")
        if item.reuse_backlot_id:
            note.append(f"reuse {item.reuse_backlot_id}")
        if item.user_reference:
            note.append("user reference")
        if item.search_name:
            note.append(f"search '{item.search_name}'")
        rows.append((item.id, item.name, item.category, size, confidence, ", ".join(note)))
    return simple_table("Inventory", ("id", "name", "category", "size (m)", "confidence", "notes"), rows)


def scorecard_table(card: ScoreCard, title: str = "Rubric") -> Table:
    rows = []
    for entry in card.entries:
        verdict = Text(entry.status, style={"passed": "green", "failed": "red", "unassessed": "yellow"}[entry.status])
        score = "-" if entry.score is None else f"{entry.score:.2f}"
        rows.append((entry.name, f"{entry.weight:g}", score, verdict, entry.decided_by, " / ".join(entry.notes)[:160]))
    table = simple_table(f"{title}: {card.overall:.2f} ({'passed' if card.passed else 'not passed'})", ("criterion", "weight", "score", "", "by", "notes"), rows)
    return table


def backlot_table(entries: Sequence[BacklotEntry]) -> Table:
    rows = [
        (e.id, e.name, e.category, e.style, f"v{e.version}", e.usd_material_mode,
         "-" if e.usd_roundtrip_score is None else f"{e.usd_roundtrip_score:.2f}")
        for e in entries
    ]
    return simple_table("Backlot", ("id", "name", "category", "style", "version", "usd materials", "round trip"), rows)


def search_table(hits: Sequence[SearchHit]) -> Table:
    rows = [(f"{h.similarity:.3f}", h.entry.id, h.entry.name, h.entry.category, h.entry.description[:80]) for h in hits]
    return simple_table("Backlot search", ("similarity", "id", "name", "category", "description"), rows)
