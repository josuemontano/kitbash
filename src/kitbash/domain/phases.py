"""Pipeline phases and their fixed order."""

from enum import StrEnum


class PhaseName(StrEnum):
    BREAKDOWN = "breakdown"
    MODELLING = "modelling"
    LAYOUT = "layout"
    ASSEMBLY = "assembly"

    @property
    def index(self) -> int:
        return PHASE_ORDER.index(self)

    @property
    def dirname(self) -> str:
        return f"{self.index + 1:02d}_{self.value}"

    @property
    def has_critic_loop(self) -> bool:
        return self is not PhaseName.ASSEMBLY

    def and_later(self) -> tuple[PhaseName, ...]:
        return PHASE_ORDER[self.index :]


PHASE_ORDER: tuple[PhaseName, ...] = tuple(PhaseName)


class PhaseStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_GATE = "awaiting_gate"
    DONE = "done"
