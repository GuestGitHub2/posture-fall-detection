"""Output from a per-person hysteresis state machine."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StateResult:
    state: str
    fall_event: bool = False
    recovered_event: bool = False
