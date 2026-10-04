"""Evaluate held-out skeleton predictions with per-class and event-aware metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from training.dataset import fall_windows, load_records, normalize_keypoints


def classification_metrics(
    truth: list[str], predictions: list[str], classes: list[str]
) -> dict[str, Any]:
    if len(truth) != len(predictions) or not truth:
        raise ValueError("Expected non-empty, equally-sized truth and prediction lists")
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for actual, predicted in zip(truth, predictions, strict=True):
        if actual not in classes or predicted not in classes:
            raise ValueError(f"Labels {actual!r}, {predicted!r} are outside class order {classes}")
        matrix[classes.index(actual), classes.index(predicted)] += 1
    per_class = {}
    for index, label in enumerate(classes):
        tp = int(matrix[index, index])
        fp, fn = int(matrix[:, index].sum()) - tp, int(matrix[index].sum()) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": int(matrix[index].sum()),
        }
    result = {
        "classes": classes,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "macro_f1": float(np.mean([row["f1"] for row in per_class.values()])),
        "accuracy": float(np.trace(matrix) / matrix.sum()),
    }
    if classes == ["normal", "fall"]:
        tn, fp, fn, tp = (int(x) for x in matrix.ravel())
        result.update(
            sensitivity=tp / (tp + fn) if tp + fn else None,
            specificity=tn / (tn + fp) if tn + fp else None,
            precision=per_class["fall"]["precision"],
            f1=per_class["fall"]["f1"],
        )
    return result


def fall_sequence_metrics(
    rows: list[dict[str, Any]], cooldown_seconds: float = 10.0, max_gap_seconds: float = 1.0
) -> dict[str, Any]:
    """Count alarm episodes and onset latency; never assume missing periods are observed."""
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["sequence_id"]), int(row.get("track_id", 0))), []).append(row)
    per_sequence = []
    false_alarms, observed_normal_seconds, latencies = 0, 0.0, []
    normal_intervals: dict[str, list[tuple[float, float]]] = {}
    sequence_tp = sequence_fn = sequence_tn = sequence_fp = 0
    for (sequence, track), samples in groups.items():
        samples.sort(key=lambda row: float(row["timestamp"]))
        truth_fall = any(row["label"] == "fall" for row in samples)
        true_alarm, false_alarm_count = False, 0
        last_alarm = -float("inf")
        previous_prediction = "normal"
        onset_values = [
            float(row["fall_onset"]) for row in samples if row.get("fall_onset") is not None
        ]
        onset = min(onset_values) if onset_values else None
        first_detection = None
        normal_duration = 0.0
        for index, row in enumerate(samples):
            timestamp = float(row["timestamp"])
            predicted = row["prediction"]
            if index:
                previous = samples[index - 1]
                delta = timestamp - float(previous["timestamp"])
                if (
                    0 < delta <= max_gap_seconds
                    and previous["label"] == "normal"
                    and row["label"] == "normal"
                ):
                    normal_duration += delta
                    normal_intervals.setdefault(sequence, []).append(
                        (float(previous["timestamp"]), timestamp)
                    )
            rising = predicted == "fall" and previous_prediction != "fall"
            if rising and timestamp - last_alarm >= cooldown_seconds:
                last_alarm = timestamp
                if row["label"] == "fall":
                    true_alarm = True
                    first_detection = timestamp if first_detection is None else first_detection
                else:
                    false_alarm_count += 1
            previous_prediction = predicted
        latency = (
            max(0.0, first_detection - onset)
            if first_detection is not None and onset is not None
            else None
        )
        if latency is not None:
            latencies.append(latency)
        false_alarms += false_alarm_count
        sequence_tp += int(truth_fall and true_alarm)
        sequence_fn += int(truth_fall and not true_alarm)
        sequence_fp += int(not truth_fall and false_alarm_count > 0)
        sequence_tn += int(not truth_fall and false_alarm_count == 0)
        per_sequence.append(
            {
                "sequence_id": sequence,
                "track_id": track,
                "contains_fall": truth_fall,
                "fall_detected": true_alarm,
                "false_alarms": false_alarm_count,
                "observed_normal_seconds": normal_duration,
                "detection_latency_seconds": latency,
            }
        )
    # Concurrent tracks share recording time: merge intervals instead of reporting person-hours.
    for intervals in normal_intervals.values():
        merged: list[list[float]] = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        observed_normal_seconds += sum(end - start for start, end in merged)
    return {
        "per_sequence": per_sequence,
        "false_alarms": false_alarms,
        "observed_normal_hours": observed_normal_seconds / 3600.0,
        "false_alarms_per_hour": false_alarms / (observed_normal_seconds / 3600.0)
        if observed_normal_seconds
        else None,
        "mean_detection_latency_seconds": float(np.mean(latencies)) if latencies else None,
        "sequence_sensitivity": sequence_tp / (sequence_tp + sequence_fn)
        if sequence_tp + sequence_fn
        else None,
        "sequence_specificity": sequence_tn / (sequence_tn + sequence_fp)
        if sequence_tn + sequence_fp
        else None,
        "event_protocol": "rising fall predictions, cooldown, union of observed normal recording intervals (concurrent tracks counted once)",
    }


def predict_checkpoint(
    data_path: Path, checkpoint_path: Path, mode: str, threshold: float
) -> tuple[list[dict[str, Any]], list[str]]:
    import torch

    from training.models import build_model

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    if metadata["kind"] != mode:
        raise ValueError(f"Checkpoint kind {metadata['kind']} does not match --mode {mode}")
    model = build_model(metadata)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    records = load_records(data_path)
    if mode == "posture":
        records = [record for record in records if record.label in metadata["classes"]]
        if not records:
            raise ValueError("No matching posture labels")
        values = np.stack(
            [normalize_keypoints(record.keypoints, bbox=record.bbox) for record in records]
        )
        labels = [record.label for record in records]
        rows = [
            {
                "sequence_id": record.sequence_id,
                "track_id": record.track_id,
                "timestamp": record.timestamp,
            }
            for record in records
        ]
    else:
        values, targets, rows = fall_windows(
            records, metadata["samples"], metadata["history_seconds"], 0.25
        )
        values = values.transpose(0, 3, 1, 2)[..., None]
        labels = [metadata["classes"][target] for target in targets]
    probability_batches = []
    with torch.inference_mode():
        for first in range(0, len(values), 128):
            probability_batches.append(
                torch.softmax(
                    model(torch.from_numpy(np.ascontiguousarray(values[first : first + 128]))), 1
                ).numpy()
            )
    probabilities = np.concatenate(probability_batches)
    for row, label, probability in zip(rows, labels, probabilities, strict=True):
        index = int(probability.argmax()) if mode == "posture" else int(probability[1] >= threshold)
        row.update(
            label=label, prediction=metadata["classes"][index], probabilities=probability.tolist()
        )
    return rows, metadata["classes"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("posture", "fall"), required=True)
    parser.add_argument(
        "--predictions",
        type=Path,
        help="JSONL label,prediction,sequence_id,track_id,timestamp,fall_onset",
    )
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--max-gap-seconds", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation.json"))
    args = parser.parse_args()
    try:
        if args.predictions:
            rows = [
                json.loads(line)
                for line in args.predictions.read_text().splitlines()
                if line.strip()
            ]
            classes = args.classes or (
                ["normal", "fall"]
                if args.mode == "fall"
                else sorted({row["label"] for row in rows})
            )
        elif args.data and args.checkpoint:
            rows, classes = predict_checkpoint(
                args.data, args.checkpoint, args.mode, args.threshold
            )
        else:
            parser.error("Supply --predictions OR both --data and --checkpoint")
        result = classification_metrics(
            [row["label"] for row in rows], [row["prediction"] for row in rows], classes
        )
        if args.mode == "fall":
            result.update(fall_sequence_metrics(rows, args.cooldown_seconds, args.max_gap_seconds))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
