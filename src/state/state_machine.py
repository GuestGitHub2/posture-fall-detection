"""Separate posture smoothing from temporal fall latching and recovery."""

from __future__ import annotations

from collections import deque

from src.fall.detector import FallResult
from src.posture.classifier import PostureResult
from src.state.person_state import StateResult

STATE_DEFAULTS = {
    "smoothing_frames": 5,
    "posture_confirmation_frames": 3,
    "posture_hold_seconds": 0.12,
    "posture_confidence_threshold": 0.4,
    "possible_threshold": 0.55,
    "confirmed_threshold": 0.8,
    "confirmation_frames": 3,
    "possible_confirmation_frames": 2,
    "max_prediction_gap_seconds": 0.5,
    "recovery_seconds": 3.0,
    "recovery_postures": ["standing", "sitting", "kneeling"],
    "event_cooldown_seconds": 10.0,
    "missing_unknown_seconds": 0.5,
}


class PersonStateMachine:
    """Instantiate one machine per track; no fall is inferred from static posture."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = STATE_DEFAULTS | (config or {})
        self.state = "UNKNOWN"
        self._posture = "UNKNOWN"
        self._votes: deque[str] = deque(maxlen=self.config["smoothing_frames"])
        self._fall_count = self._possible_count = 0
        self._fallen = False
        self._recovery_since: float | None = None
        self._last_update: float | None = None
        self._last_event = float("-inf")
        self._vote_since: float | None = None
        self._last_vote = "UNKNOWN"

    def missing(self, timestamp: float) -> StateResult:
        if (
            self._last_update is not None
            and timestamp - self._last_update > self.config["missing_unknown_seconds"]
        ):
            self._fall_count = self._possible_count = 0
            self._recovery_since = None
            self._votes.clear()
            self._vote_since = None
            self.state = "FALLEN" if self._fallen else "UNKNOWN"
        return StateResult(self.state)

    def update(self, posture: PostureResult, fall: FallResult, timestamp: float) -> StateResult:
        config = self.config
        if self._last_update is not None and (
            timestamp <= self._last_update
            or timestamp - self._last_update > config["max_prediction_gap_seconds"]
        ):
            self._fall_count = self._possible_count = 0
            self._votes.clear()
            self._recovery_since = None
            self._vote_since = None
        self._last_update = timestamp
        vote = (
            posture.label.upper()
            if posture.confidence >= config["posture_confidence_threshold"]
            else "UNKNOWN"
        )
        if vote != self._last_vote or self._vote_since is None:
            self._vote_since = timestamp
        self._last_vote = vote
        self._votes.append(vote)
        if (
            len(self._votes) >= config["posture_confirmation_frames"]
            and list(self._votes)[-config["posture_confirmation_frames"] :]
            == [vote] * config["posture_confirmation_frames"]
            and timestamp - self._vote_since >= config["posture_hold_seconds"]
        ):
            self._posture = vote
        confirmed = fall.status == "fall" and fall.probability >= config["confirmed_threshold"]
        possible = (
            fall.status in {"fall", "possible_fall"}
            and fall.probability >= config["possible_threshold"]
        )
        self._fall_count = self._fall_count + 1 if confirmed else 0
        self._possible_count = self._possible_count + 1 if possible else 0
        event = recovered = False
        if not self._fallen and self._fall_count >= config["confirmation_frames"]:
            self._fallen = True
            self._recovery_since = None
            if timestamp - self._last_event >= config["event_cooldown_seconds"]:
                event = True
                self._last_event = timestamp
        if self._fallen:
            upright = (
                posture.label in config["recovery_postures"]
                and posture.confidence >= config["posture_confidence_threshold"]
                and not possible
            )
            if upright:
                if self._recovery_since is None:
                    self._recovery_since = timestamp
                self.state = "RECOVERING"
                if timestamp - self._recovery_since >= config["recovery_seconds"]:
                    self._fallen = False
                    self.state = self._posture
                    self._fall_count = self._possible_count = 0
                    self._recovery_since = None
                    recovered = True
            else:
                self._recovery_since = None
                self.state = "FALLEN"
        elif self._possible_count >= config["possible_confirmation_frames"]:
            self.state = "FALLING"
        else:
            self.state = self._posture
        return StateResult(self.state, event, recovered)
