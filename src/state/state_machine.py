"""Timestamp-duration fall confirmation, episode latching and upright rearming."""

from __future__ import annotations

import logging
import math
from collections import deque

from src.fall.detector import FallResult
from src.posture.classifier import PostureResult
from src.state.person_state import StateResult

STATE_DEFAULTS = {
    "smoothing_frames": 5,
    "posture_confirmation_frames": 3,
    "posture_hold_seconds": 0.12,
    "posture_confidence_threshold": 0.4,
    "minimum_pose_quality": 0.4,
    "possible_threshold": 0.55,
    "confirmed_threshold": 0.8,
    "fall_confirmation_seconds": 0.2,
    "possible_confirmation_seconds": 0.08,
    "max_prediction_gap_seconds": 0.5,
    "recovery_seconds": 3.0,
    "recovery_postures": ["standing", "sitting", "kneeling"],
    "rearm_seconds": 1.0,
    "rearm_postures": ["standing", "sitting"],
    "missing_unknown_seconds": 0.5,
}
LOG = logging.getLogger(__name__)


class PersonStateMachine:
    """One event per episode; another requires recovery plus a stable upright period."""

    def __init__(self, config: dict | None = None) -> None:
        supplied = config or {}
        self.config = STATE_DEFAULTS | supplied
        # Legacy overrides are translated once at the historical 15 Hz reference,
        # never counted at the current FPS. New duration keys take precedence.
        if "confirmation_frames" in supplied and (
            "fall_confirmation_seconds" not in supplied
            or (
                supplied["confirmation_frames"] != 3
                and supplied["fall_confirmation_seconds"]
                == STATE_DEFAULTS["fall_confirmation_seconds"]
            )
        ):
            self.config["fall_confirmation_seconds"] = max(
                0, (supplied["confirmation_frames"] - 1) / 15
            )
            LOG.warning(
                "confirmation_frames is deprecated; interpreted as %.3f seconds",
                self.config["fall_confirmation_seconds"],
            )
        if (
            "possible_confirmation_frames" in supplied
            and "possible_confirmation_seconds" not in supplied
        ):
            self.config["possible_confirmation_seconds"] = max(
                0, (supplied["possible_confirmation_frames"] - 1) / 15
            )
        self.state = "UNKNOWN"
        self._posture = "UNKNOWN"
        self._votes: deque[str] = deque(maxlen=self.config["smoothing_frames"])
        self._fall_duration = self._possible_duration = 0.0
        self._confirmed_previous = self._possible_previous = False
        self._fallen = False
        self._armed = True
        self._arm_hold_seconds = self.config["rearm_seconds"]
        self._recovery_since: float | None = None
        self._rearm_since: float | None = None
        self._last_update: float | None = None
        self._last_valid: float | None = None
        self._interrupted = False
        self._vote_since: float | None = None
        self._last_vote = "UNKNOWN"

    @property
    def armed(self) -> bool:
        return self._armed

    def disarm(self, hold_seconds: float | None = None) -> None:
        self._armed = False
        self._rearm_since = None
        self._arm_hold_seconds = (
            self.config["rearm_seconds"] if hold_seconds is None else hold_seconds
        )

    def _reset_predictions(self) -> None:
        self._fall_duration = self._possible_duration = 0.0
        self._confirmed_previous = self._possible_previous = False
        self._last_valid = None
        self._recovery_since = self._rearm_since = None
        self._votes.clear()
        self._vote_since = None

    def invalidate_identity(self) -> None:
        """No posture/fall latch or temporal evidence may cross an uncertain identity."""
        self._reset_predictions()
        self._fallen = False
        self._posture = self.state = "UNKNOWN"
        self.disarm()

    def missing(self, timestamp: float) -> StateResult:
        self._interrupted = True
        self._recovery_since = self._rearm_since = None
        if (
            self._last_valid is not None
            and timestamp - self._last_valid > self.config["missing_unknown_seconds"]
        ):
            self._reset_predictions()
            self.state = "FALLEN" if self._fallen else "UNKNOWN"
        return StateResult(self.state)

    def update(self, posture: PostureResult, fall: FallResult, timestamp: float) -> StateResult:
        c = self.config
        if not math.isfinite(timestamp):
            raise ValueError("State timestamp must be finite")
        if self._last_update is not None and (
            timestamp <= self._last_update
            or timestamp - self._last_update > c["max_prediction_gap_seconds"]
        ):
            self._reset_predictions()
            self._interrupted = True
        self._last_update = timestamp
        quality_ok = float(posture.pose_quality or 0) >= c["minimum_pose_quality"]
        vote = (
            posture.label.upper()
            if quality_ok and posture.confidence >= c["posture_confidence_threshold"]
            else "UNKNOWN"
        )
        if vote != self._last_vote or self._vote_since is None:
            self._vote_since = timestamp
        self._last_vote = vote
        self._votes.append(vote)
        if (
            len(self._votes) >= c["posture_confirmation_frames"]
            and list(self._votes)[-c["posture_confirmation_frames"] :]
            == [vote] * c["posture_confirmation_frames"]
            and timestamp - self._vote_since + 1e-9 >= c["posture_hold_seconds"]
        ):
            self._posture = vote
        if not quality_ok or fall.features.get("measurement_valid", 1.0) == 0:
            # An unusable floor/motion measurement must not erase a confident
            # posture vote. Only fall confirmation and recovery are suspended.
            self._interrupted = True
            self._recovery_since = self._rearm_since = None
            if (
                self._last_valid is not None
                and timestamp - self._last_valid > c["max_prediction_gap_seconds"]
            ):
                self._fall_duration = self._possible_duration = 0.0
                self._confirmed_previous = self._possible_previous = False
                self._last_valid = None
            self.state = "FALLEN" if self._fallen else self._posture
            return StateResult(self.state)
        delta = (
            timestamp - self._last_valid
            if self._last_valid is not None and not self._interrupted
            else 0.0
        )
        self._last_valid = timestamp
        self._interrupted = False
        confirmed = fall.status == "fall" and fall.probability >= c["confirmed_threshold"]
        possible = (
            fall.status in {"fall", "possible_fall"} and fall.probability >= c["possible_threshold"]
        )
        self._fall_duration = (
            self._fall_duration + delta if confirmed and self._confirmed_previous else 0.0
        )
        self._possible_duration = (
            self._possible_duration + delta if possible and self._possible_previous else 0.0
        )
        self._confirmed_previous, self._possible_previous = confirmed, possible
        event = recovered = False
        if (
            not self._fallen
            and confirmed
            and self._fall_duration + 1e-9 >= c["fall_confirmation_seconds"]
        ):
            self._fallen = True
            self._recovery_since = None
            event = self._armed
            self.disarm()
        if self._fallen:
            upright = (
                posture.label in c["recovery_postures"]
                and posture.confidence >= c["posture_confidence_threshold"]
                and not possible
            )
            if upright:
                if self._recovery_since is None:
                    self._recovery_since = timestamp
                self.state = "RECOVERING"
                if timestamp - self._recovery_since + 1e-9 >= c["recovery_seconds"]:
                    self._fallen = False
                    self.state = self._posture
                    self._fall_duration = self._possible_duration = 0.0
                    self._confirmed_previous = self._possible_previous = False
                    self._recovery_since = None
                    self._rearm_since = timestamp
                    recovered = True
            else:
                self._recovery_since = None
                self.state = "FALLEN"
        elif possible and self._possible_duration + 1e-9 >= c["possible_confirmation_seconds"]:
            self.state = "FALLING"
            self._rearm_since = None
        else:
            self.state = self._posture
        if not self._fallen and not self._armed:
            upright = (
                posture.label in c["rearm_postures"]
                and posture.confidence >= c["posture_confidence_threshold"]
                and not possible
            )
            if upright:
                if self._rearm_since is None:
                    self._rearm_since = timestamp
                if timestamp - self._rearm_since + 1e-9 >= self._arm_hold_seconds:
                    self._armed = True
            else:
                self._rearm_since = None
        return StateResult(self.state, event, recovered)
