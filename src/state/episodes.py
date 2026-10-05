"""Alert deduplication across uncertain ID loss, without transferring pose histories."""

from dataclasses import dataclass

import numpy as np

from src.pose.pose_types import Pose
from src.tracking.matching import center_distance, iou

EPISODE_DEFAULTS = {
    "episode_retention_seconds": 60.0,
    "episode_match_distance": 0.8,
    "episode_min_iou": 0.05,
}


@dataclass
class Episode:
    episode_id: int
    track_id: int
    pose: Pose
    last_seen: float
    history_epoch: int = 0
    owner_visible: bool = True
    rearmed: bool = False


class EpisodeRegistry:
    """Quarantine uncertain reacquisitions, never merge people with continuous IDs.

    Spatial proximity is only a conservative duplicate-alert guard for new or
    identity-invalidated tracks whose former owner is lost. It never supplies
    skeleton history, fall probabilities or a person's state to another ID.
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = EPISODE_DEFAULTS | (config or {})
        self.episodes: dict[int, Episode] = {}
        self._quarantined: dict[int, set[int]] = {}
        self._next_id = 1

    def _matches(self, episode: Episode, track_id: int, pose: Pose) -> bool:
        return episode.track_id == track_id or (
            not episode.owner_visible
            and center_distance(episode.pose, pose, np.zeros(2))
            <= self.config["episode_match_distance"]
            and iou(episode.pose.bbox, pose.bbox) >= self.config["episode_min_iou"]
        )

    def prune(self, timestamp: float) -> None:
        self.episodes = {
            key: episode
            for key, episode in self.episodes.items()
            if 0 <= timestamp - episode.last_seen <= self.config["episode_retention_seconds"]
        }
        self._quarantined = {
            track: ids & self.episodes.keys()
            for track, ids in self._quarantined.items()
            if ids & self.episodes.keys()
        }

    def mark_owners(self, observed_epochs: dict[int, int]) -> None:
        for episode in self.episodes.values():
            episode.owner_visible = observed_epochs.get(episode.track_id) == episode.history_epoch

    def quarantine(self, track_id: int, pose: Pose) -> bool:
        ids = {
            episode.episode_id
            for episode in self.episodes.values()
            if not episode.rearmed and self._matches(episode, track_id, pose)
        }
        if ids:
            self._quarantined[track_id] = ids
        return bool(ids)

    def observe(self, track_id: int, pose: Pose, timestamp: float, rearmed: bool) -> None:
        for episode in self.episodes.values():
            owner = episode.track_id == track_id and episode.owner_visible
            guarded = episode.episode_id in self._quarantined.get(track_id, set())
            if owner:
                episode.pose, episode.last_seen = pose, timestamp
            if rearmed and (owner or guarded):
                episode.rearmed = True
        if rearmed:
            self._quarantined.pop(track_id, None)

    def confirm(
        self, track_id: int, pose: Pose, timestamp: float, history_epoch: int = 0
    ) -> int | None:
        if any(
            not episode.rearmed
            and (
                episode.track_id == track_id
                or episode.episode_id in self._quarantined.get(track_id, set())
            )
            for episode in self.episodes.values()
        ):
            return None
        episode = Episode(self._next_id, track_id, pose, timestamp, history_epoch)
        self.episodes[episode.episode_id] = episode
        self._next_id += 1
        return episode.episode_id

    def episode_for(self, track_id: int) -> int | None:
        ids = [
            episode.episode_id
            for episode in self.episodes.values()
            if not episode.rearmed
            and (
                episode.track_id == track_id
                or episode.episode_id in self._quarantined.get(track_id, set())
            )
        ]
        return max(ids) if ids else None
