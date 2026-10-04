"""Train a tiny normalized-skeleton posture MLP using grouped validation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from training.dataset import POSTURE_CLASSES, grouped_split, load_records, normalize_keypoints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("models/posture_mlp.pt"))
    parser.add_argument("--export", type=Path)
    parser.add_argument("--classes", nargs="+", default=list(POSTURE_CLASSES))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        from training.train_common import train

        records = [row for row in load_records(args.data) if row.label in args.classes]
        train_records, validation_records = grouped_split(
            records, args.validation_fraction, args.seed
        )

        def arrays(rows):
            return (
                np.stack([normalize_keypoints(row.keypoints, bbox=row.bbox) for row in rows]),
                np.array([args.classes.index(row.label) for row in rows], dtype=np.int64),
            )

        data, labels = arrays(train_records)
        validation_data, validation_labels = arrays(validation_records)
        metadata = {
            "kind": "posture",
            "classes": args.classes,
            "hidden": 64,
            "normalization": "hip_center_torso_plus_leg_length",
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
