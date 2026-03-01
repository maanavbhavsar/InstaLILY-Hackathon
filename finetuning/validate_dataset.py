from __future__ import annotations

"""
Pre-training sanity checks for the GRID lip-reading dataset.

Checks:
- All mosaic images referenced in JSONL files exist and are 400x240
- All transcriptions are non-empty and follow GRID grammar (6 words)
- Split sizes are reasonable
- No duplicate samples across splits

Usage:
    python -m finetuning.validate_dataset
    python -m finetuning.validate_dataset --dataset-dir data/
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import cv2

from .config import (
    DATASET_DIR,
    MOSAIC_CELL_WIDTH,
    MOSAIC_CELL_HEIGHT,
    MOSAIC_GRID_ROWS,
    MOSAIC_GRID_COLS,
    GRID_COMMANDS,
    GRID_COLORS,
    GRID_PREPOSITIONS,
    GRID_LETTERS,
    GRID_DIGITS,
    GRID_ADVERBS,
)

logger = logging.getLogger(__name__)

EXPECTED_MOSAIC_WIDTH = MOSAIC_GRID_COLS * MOSAIC_CELL_WIDTH  # 400
EXPECTED_MOSAIC_HEIGHT = MOSAIC_GRID_ROWS * MOSAIC_CELL_HEIGHT  # 240

# Build vocabulary sets for validation
VOCAB_SLOTS = [
    set(GRID_COMMANDS),
    set(GRID_COLORS),
    set(GRID_PREPOSITIONS),
    set(GRID_LETTERS),
    set(GRID_DIGITS),
    set(GRID_ADVERBS),
]


def load_jsonl(path: Path) -> list[dict]:
    """Load samples from a JSONL file."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_sample(sample: dict, check_images: bool = True) -> list[str]:
    """Validate a single sample. Returns list of error messages."""
    errors = []

    # Check structure
    if "messages" not in sample:
        errors.append("Missing 'messages' key")
        return errors

    msgs = sample["messages"]
    if len(msgs) != 2:
        errors.append(f"Expected 2 messages, got {len(msgs)}")
        return errors

    if msgs[0]["role"] != "user" or msgs[1]["role"] != "assistant":
        errors.append("Wrong message roles")

    # Check user content has image + text
    user_content = msgs[0].get("content", [])
    if not isinstance(user_content, list) or len(user_content) < 2:
        errors.append("User content should be list with image + text")
        return errors

    image_entry = user_content[0]
    if image_entry.get("type") != "image" or "image" not in image_entry:
        errors.append("First content item should be image")

    # Check image file exists and has correct dimensions
    if check_images and "image" in image_entry:
        img_path = Path(image_entry["image"])
        if not img_path.exists():
            errors.append(f"Image not found: {img_path}")
        else:
            img = cv2.imread(str(img_path))
            if img is None:
                errors.append(f"Cannot read image: {img_path}")
            else:
                h, w = img.shape[:2]
                if (w, h) != (EXPECTED_MOSAIC_WIDTH, EXPECTED_MOSAIC_HEIGHT):
                    errors.append(
                        f"Wrong image dims: {w}x{h} "
                        f"(expected {EXPECTED_MOSAIC_WIDTH}x{EXPECTED_MOSAIC_HEIGHT})"
                    )

    # Check transcription
    transcription = msgs[1].get("content", "")
    if not transcription.strip():
        errors.append("Empty transcription")
    else:
        words = transcription.strip().split()
        if len(words) != 6:
            errors.append(
                f"Expected 6 words in transcription, got {len(words)}: '{transcription}'"
            )
        else:
            # Check each word is in the right vocabulary slot
            for i, (word, valid_set) in enumerate(zip(words, VOCAB_SLOTS)):
                if word not in valid_set:
                    errors.append(
                        f"Word '{word}' at position {i} not in GRID vocabulary"
                    )

    return errors


def validate_dataset(
    dataset_dir: Path = DATASET_DIR,
    check_images: bool = True,
) -> bool:
    """
    Run all validation checks on the dataset.
    Returns True if all checks pass.
    """
    all_ok = True
    all_image_paths: list[str] = []

    for split in ["train", "val", "test"]:
        path = dataset_dir / f"grid_dataset_{split}.jsonl"
        samples = load_jsonl(path)

        if not samples:
            logger.warning(f"{split}: NO SAMPLES FOUND at {path}")
            if split == "train":
                all_ok = False
            continue

        print(f"\n=== {split.upper()} ({len(samples)} samples) ===")

        error_count = 0
        image_paths = []

        for idx, sample in enumerate(samples):
            errors = validate_sample(sample, check_images=check_images)
            if errors:
                error_count += 1
                if error_count <= 5:  # Show first 5 errors
                    print(f"  Sample {idx}: {'; '.join(errors)}")

            # Collect image paths for duplicate check
            content = sample.get("messages", [{}])[0].get("content", [])
            if isinstance(content, list) and content:
                img = content[0].get("image", "")
                if img:
                    image_paths.append(img)

        if error_count > 5:
            print(f"  ... and {error_count - 5} more errors")

        if error_count > 0:
            print(f"  ERRORS: {error_count}/{len(samples)} samples have issues")
            all_ok = False
        else:
            print(f"  OK: all {len(samples)} samples valid")

        all_image_paths.extend(image_paths)

        # Check for within-split duplicates
        dupes = [p for p, c in Counter(image_paths).items() if c > 1]
        if dupes:
            print(f"  WARNING: {len(dupes)} duplicate images within {split}")
            all_ok = False

    # Check for cross-split duplicates
    cross_dupes = [p for p, c in Counter(all_image_paths).items() if c > 1]
    if cross_dupes:
        print(f"\nWARNING: {len(cross_dupes)} images appear in multiple splits!")
        for p in cross_dupes[:3]:
            print(f"  {p}")
        all_ok = False
    else:
        print(f"\nNo cross-split duplicates found")

    # Summary
    print(f"\nTotal samples: {len(all_image_paths)}")
    print(f"Validation: {'PASSED' if all_ok else 'FAILED'}")

    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Validate GRID lip-reading dataset"
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Skip image file checks (faster)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    ok = validate_dataset(
        dataset_dir=args.dataset_dir,
        check_images=not args.skip_images,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
