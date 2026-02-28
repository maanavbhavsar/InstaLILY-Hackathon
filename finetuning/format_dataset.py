from __future__ import annotations

"""
Format GRID lip-reading data into JSONL files for SFTTrainer.

Pairs mosaic images with ground truth transcriptions and splits into
train/val/test JSONL files.

Each line:
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "<path_to_mosaic>"},
        {"type": "text", "text": "<prompt>"}
      ]
    },
    {
      "role": "assistant",
      "content": "<transcription>"
    }
  ]
}

Usage:
    python -m finetuning.format_dataset --speakers s1
    python -m finetuning.format_dataset --speakers s1 s2 s3 s4 s5
"""
import argparse
import json
import logging
import random
from pathlib import Path

from .config import (
    MOSAIC_DIR,
    DATASET_DIR,
    DEFAULT_SPEAKERS,
    LIP_READING_PROMPT,
    TRAIN_RATIO,
    VAL_RATIO,
    SPLIT_SEED,
)
from .parse_alignments import parse_all_alignments

logger = logging.getLogger(__name__)


def build_samples(
    speakers: list[str],
    mosaic_root: Path = MOSAIC_DIR,
) -> list[dict]:
    """
    Build list of (mosaic_path, transcription) samples.
    Only includes samples where the mosaic exists.
    """
    alignments = parse_all_alignments(speakers)
    logger.info(f"Loaded {len(alignments)} alignments")

    samples = []
    missing = 0

    for key, info in alignments.items():
        speaker, video_id = key.split("/")
        mosaic_path = mosaic_root / speaker / f"{video_id}.jpg"

        if not mosaic_path.exists():
            missing += 1
            continue

        if not info.transcription.strip():
            continue

        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": str(mosaic_path)},
                        {"type": "text", "text": LIP_READING_PROMPT},
                    ],
                },
                {
                    "role": "assistant",
                    "content": info.transcription,
                },
            ]
        }
        samples.append(sample)

    if missing > 0:
        logger.info(f"Skipped {missing} samples with missing mosaics")

    logger.info(f"Built {len(samples)} total samples")
    return samples


def split_dataset(
    samples: list[dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = SPLIT_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split samples into train/val/test sets with deterministic shuffling."""
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    return train, val, test


def write_jsonl(samples: list[dict], path: Path) -> None:
    """Write samples to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(samples)} samples to {path}")


def format_dataset(
    speakers: list[str],
    mosaic_root: Path = MOSAIC_DIR,
    output_dir: Path = DATASET_DIR,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = SPLIT_SEED,
) -> tuple[int, int, int]:
    """
    Build and split the dataset, writing train/val/test JSONL files.
    Returns (n_train, n_val, n_test).
    """
    samples = build_samples(speakers, mosaic_root=mosaic_root)

    if not samples:
        logger.error("No samples found! Check that mosaics exist.")
        return 0, 0, 0

    train, val, test = split_dataset(
        samples, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed
    )

    write_jsonl(train, output_dir / "grid_dataset_train.jsonl")
    write_jsonl(val, output_dir / "grid_dataset_val.jsonl")
    write_jsonl(test, output_dir / "grid_dataset_test.jsonl")

    return len(train), len(val), len(test)


def main():
    parser = argparse.ArgumentParser(
        description="Format GRID dataset for SFTTrainer"
    )
    parser.add_argument("--speakers", nargs="+", default=DEFAULT_SPEAKERS)
    parser.add_argument("--mosaic-dir", type=Path, default=MOSAIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    n_train, n_val, n_test = format_dataset(
        speakers=args.speakers,
        mosaic_root=args.mosaic_dir,
        output_dir=args.output_dir,
    )

    print(f"Dataset formatted: train={n_train}, val={n_val}, test={n_test}")


if __name__ == "__main__":
    main()
