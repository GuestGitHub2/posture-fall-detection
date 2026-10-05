"""Train a small graph-temporal skeleton fall model with grouped validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.fall.normalization import FALL_NORMALIZATION
from training.dataset import fall_windows, grouped_split, load_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/fall_temporal.pt"))
    parser.add_argument("--export", type=Path)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--history-seconds", type=float, default=2.5)
    parser.add_argument("--max-gap-seconds", type=float, default=0.5)
    parser.add_argument("--joint-confidence", type=float, default=0.4)
    parser.add_argument("--minimum-scale-quality", type=float, default=0.6)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        from training.train_common import train

        train_records, validation_records = grouped_split(
            load_records(args.data), args.validation_fraction, args.seed
        )
        data, labels, _ = fall_windows(
            train_records,
            args.samples,
            args.history_seconds,
            args.stride_seconds,
            args.max_gap_seconds,
            args.joint_confidence,
            args.minimum_scale_quality,
        )
        validation_data, validation_labels, _ = fall_windows(
            validation_records,
            args.samples,
            args.history_seconds,
            args.stride_seconds,
            args.max_gap_seconds,
            args.joint_confidence,
            args.minimum_scale_quality,
        )
        metadata = {
            "kind": "fall",
            "classes": ["normal", "fall"],
            "channels": 32,
            "samples": args.samples,
            "history_seconds": args.history_seconds,
            "layout": "NCTVM",
            "normalization": FALL_NORMALIZATION,
            "origin": "first_reliable_hip_center",
            "scale": "median_reliable_torso_plus_leg_length",
            "minimum_joint_confidence": args.joint_confidence,
            "minimum_scale_quality": args.minimum_scale_quality,
            "max_gap_seconds": args.max_gap_seconds,
            "train_sequences": sorted({r.sequence_id for r in train_records}),
            "validation_sequences": sorted({r.sequence_id for r in validation_records}),
        }
        train(
            data,
            labels,
            validation_data,
            validation_labels,
            metadata,
            args.output,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.seed,
            args.export,
        )
    except (ValueError, OSError, ImportError) as error:
        parser.exit(
            2, f"Training failed: {error}\nInstall training extras: pip install -e '.[training]'\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
