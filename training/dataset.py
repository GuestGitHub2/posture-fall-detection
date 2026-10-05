"""Validated skeleton records, grouped splits and confidence-aware augmentation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.fall.normalization import (
    FALL_NORMALIZATION,
    interpolate_skeletons,
    normalize_fall_sequence,
)
from src.pose.pose_types import Pose

COCO_FLIP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
POSTURE_CLASSES = ("standing", "sitting", "bending", "lying", "squatting", "kneeling", "unknown")


@dataclass(frozen=True)
class SkeletonRecord:
    sequence_id: str
    frame_number: int
    track_id: int
    timestamp: float
    keypoints: np.ndarray
    label: str
    subject_id: str | None = None
    fall_onset: float | None = None
    bbox: np.ndarray | None = None
    history_epoch: int = 0


def load_records(path: str | Path) -> list[SkeletonRecord]:
    """Read JSONL files; report malformed records with their file and line number."""
    source = Path(path)
    files = sorted(source.rglob("*.jsonl")) if source.is_dir() else [source]
    if not files:
        raise ValueError(f"No JSONL skeleton files found in {source}")
    records = []
    for file in files:
        with file.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    points = np.asarray(row["keypoints"], dtype=np.float32)
                    if points.shape != (17, 3) or not np.isfinite(points).all():
                        raise ValueError("keypoints must be finite [17,3] x/y/confidence")
                    if np.any((points[:, 2] < 0) | (points[:, 2] > 1)):
                        raise ValueError("joint confidence must lie in [0,1]")
                    timestamp = float(row["timestamp"])
                    if not np.isfinite(timestamp):
                        raise ValueError("timestamp must be finite")
                    bbox = (
                        np.asarray(row["bbox"], dtype=np.float32)
                        if row.get("bbox") is not None
                        else None
                    )
                    if bbox is not None and (
                        bbox.shape != (4,)
                        or not np.isfinite(bbox).all()
                        or np.any(bbox[2:] <= bbox[:2])
                    ):
                        raise ValueError("bbox must be finite positive-area [x1,y1,x2,y2]")
                    records.append(
                        SkeletonRecord(
                            str(row["sequence_id"]),
                            int(row["frame_number"]),
                            int(row["track_id"]),
                            timestamp,
                            points,
                            str(row.get("label", "unknown")),
                            str(row["subject_id"]) if row.get("subject_id") is not None else None,
                            float(row["fall_onset"]) if row.get("fall_onset") is not None else None,
                            bbox,
                            int(row.get("history_epoch", 0)),
                        )
                    )
                except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
                    raise ValueError(f"{file}:{number}: {error}") from error
    return records


def grouped_split(
    records: list[SkeletonRecord], validation_fraction: float = 0.2, seed: int = 42
) -> tuple[list[SkeletonRecord], list[SkeletonRecord]]:
    """Keep an entire subject (or an unannotated sequence) in exactly one split."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")

    parents: dict[str, str] = {}

    def find(node: str) -> str:
        parents.setdefault(node, node)
        if parents[node] != node:
            parents[node] = find(parents[node])
        return parents[node]

    # Connect recordings and participants. A shared multi-person video cannot
    # leak across splits even if its tracks have distinct participant annotations.
    for record in records:
        sequence = find(f"sequence:{record.sequence_id}")
        if record.subject_id:
            subject = find(f"subject:{record.subject_id}")
            if sequence != subject:
                parents[max(sequence, subject)] = min(sequence, subject)

    def key(record: SkeletonRecord) -> str:
        return find(f"sequence:{record.sequence_id}")

    groups = sorted({key(row) for row in records})
    if len(groups) < 2:
        raise ValueError(
            "At least two independent subjects/sequences are required for a grouped split"
        )
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    count = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
    validation = set(groups[:count])
    return (
        [row for row in records if key(row) not in validation],
        [row for row in records if key(row) in validation],
    )


def normalize_keypoints(
    points: np.ndarray, confidence_threshold: float = 0.3, bbox: np.ndarray | None = None
) -> np.ndarray:
    """Hip centered, translation/scale invariant COCO skeleton; invalid joints stay zero."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (17, 3):
        raise ValueError("Expected a COCO [17,3] skeleton")
    from src.pose.normalization import normalize_pose
    from src.pose.pose_types import Pose

    valid = points[:, 2] >= confidence_threshold
    if not valid.any():
        return np.zeros_like(points)
    minimum, maximum = points[valid, :2].min(axis=0), points[valid, :2].max(axis=0)
    maximum = np.maximum(maximum, minimum + 1e-3)
    normalized = normalize_pose(
        Pose(points, np.r_[minimum, maximum] if bbox is None else bbox), confidence_threshold
    ).normalized
    assert normalized is not None
    return normalized


def resample_sequence(
    points: np.ndarray,
    timestamps: np.ndarray,
    samples: int,
    start: float | None = None,
    end: float | None = None,
    max_gap_seconds: float = 0.5,
) -> np.ndarray:
    """Uniform timestamp sampling; interpolate short visible-joint gaps only."""
    if samples < 2 or len(points) != len(timestamps) or len(points) < 1:
        raise ValueError("Need matching non-empty points/timestamps and at least two samples")
    order = np.argsort(timestamps, kind="stable")
    timestamps, points = np.asarray(timestamps)[order], np.asarray(points)[order]
    timestamps, indices = np.unique(timestamps, return_index=True)
    points = points[indices]
    lo = float(timestamps[0]) if start is None else start
    hi = float(timestamps[-1]) if end is None else end
    grid = np.linspace(lo, hi, samples)
    return interpolate_skeletons(points, timestamps, grid, max_gap_seconds)


def augment(
    points: np.ndarray,
    rng: np.random.Generator,
    *,
    noise: float = 0.015,
    dropout: float = 0.06,
    flip_probability: float = 0.5,
) -> np.ndarray:
    """Augment normalized [T,17,3] or [17,3] data without reviving missing joints."""
    result = np.asarray(points, dtype=np.float32).copy()
    was_frame = result.ndim == 2
    if was_frame:
        result = result[None]
    if result.ndim != 3 or result.shape[1:] != (17, 3):
        raise ValueError("Expected [T,17,3] or [17,3]")
    if rng.random() < flip_probability:
        result = result[:, COCO_FLIP].copy()
        result[:, :, 0] *= -1
    visible = result[:, :, 2] >= 0.3
    result[:, :, :2] *= rng.uniform(0.9, 1.1)
    result[:, :, :2] += rng.normal(0, noise, result[:, :, :2].shape)
    # Re-normalized poses are translation invariant; translation is applied before extraction.
    result[:, :, 2] *= rng.uniform(0.7, 1.0, result[:, :, 2].shape)
    missing = (rng.random(visible.shape) < dropout) | ~visible
    result[missing] = 0
    if len(result) > 2:
        # Random crop/resampling gives temporal speed variation without changing target length.
        length = len(result)
        crop = max(2, round(length * rng.uniform(0.7, 1.0)))
        # Keep the labeled endpoint so a causal positive window retains its fall annotation.
        first = length - crop
        result = resample_sequence(
            result[first : first + crop], np.arange(crop), length, max_gap_seconds=1.01
        )
    return result[0] if was_frame else result


def sequence_groups(
    records: Iterable[SkeletonRecord],
) -> dict[tuple[str, int, int], list[SkeletonRecord]]:
    groups: dict[tuple[str, int, int], list[SkeletonRecord]] = {}
    for record in records:
        groups.setdefault((record.sequence_id, record.track_id, record.history_epoch), []).append(
            record
        )
    # Frame order is causal. Sorting by timestamp would hide clock regressions
    # and interleave samples from opposite sides of a discontinuity.
    return {
        key: sorted(rows, key=lambda row: (row.frame_number, row.timestamp))
        for key, rows in groups.items()
    }


def fall_windows(
    records: list[SkeletonRecord],
    samples: int = 48,
    history_seconds: float = 2.5,
    stride_seconds: float = 0.5,
    max_gap_seconds: float = 0.5,
    confidence_threshold: float = 0.3,
    minimum_scale_quality: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Create causal windows. A window is positive when its final annotation is fall."""
    windows, labels, metadata = [], [], []
    if history_seconds <= 0 or stride_seconds <= 0 or max_gap_seconds <= 0:
        raise ValueError("Fall window durations and stride must be positive")
    segments = []
    for key, group in sequence_groups(records).items():
        boundaries = (
            [0]
            + [
                index
                for index in range(1, len(group))
                if not 0 < group[index].timestamp - group[index - 1].timestamp <= max_gap_seconds
            ]
            + [len(group)]
        )
        segments.extend(
            (key, group[start:stop])
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        )
    for (sequence, track, epoch), rows in segments:
        times = np.array([row.timestamp for row in rows])
        poses = []
        for row in rows:
            visible = row.keypoints[:, 2] >= confidence_threshold
            low = row.keypoints[visible, :2].min(axis=0) if visible.any() else np.zeros(2)
            high = row.keypoints[visible, :2].max(axis=0) if visible.any() else np.ones(2)
            box = row.bbox if row.bbox is not None else np.r_[low, np.maximum(high, low + 1e-3)]
            poses.append(Pose(row.keypoints, box))
        moment = float(times[0] + history_seconds)
        while moment <= times[-1] + 1e-6:
            start = int(np.searchsorted(times, moment - history_seconds, side="left"))
            stop = int(np.searchsorted(times, moment, side="right"))
            if stop - start >= 2 and np.all(
                (np.diff(times[start:stop]) > 0) & (np.diff(times[start:stop]) <= max_gap_seconds)
            ):
                points, transform = normalize_fall_sequence(
                    poses[start:stop], confidence_threshold, minimum_scale_quality
                )
                if not transform.reliable:
                    moment += stride_seconds
                    continue
                windows.append(
                    interpolate_skeletons(
                        points,
                        times[start:stop],
                        np.linspace(moment - history_seconds, moment, samples),
                        max_gap_seconds,
                        confidence_threshold,
                    )
                )
                label = rows[stop - 1].label
                if label not in ("normal", "fall", "falling", "fallen", "recovery"):
                    raise ValueError(f"Fall label {label!r} in {sequence}; expected normal/fall")
                labels.append(int(label in ("fall", "falling", "fallen")))
                metadata.append(
                    {
                        "sequence_id": sequence,
                        "track_id": track,
                        "timestamp": moment,
                        "fall_onset": rows[stop - 1].fall_onset,
                        "history_epoch": epoch,
                        "normalization": FALL_NORMALIZATION,
                        "origin": transform.origin.tolist(),
                        "body_scale": transform.scale,
                    }
                )
            moment += stride_seconds
    if not windows:
        raise ValueError("No full temporal windows. Check durations and history_seconds")
    return np.stack(windows), np.asarray(labels, dtype=np.int64), metadata
