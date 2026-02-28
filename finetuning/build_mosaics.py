from __future__ import annotations

"""
Build 3x4 mosaic images from mouth ROI frames.

For each video:
1. Get speech boundaries from alignment (skip silence)
2. Sample 12 evenly-spaced frames from the speech region
3. Load corresponding mouth ROIs, resize to 100x80
4. Arrange in 3x4 grid -> 400x240 mosaic
5. Save as JPEG (quality 95)

Usage:
    python -m finetuning.build_mosaics --speakers s1
    python -m finetuning.build_mosaics --speakers s1 --limit 5
    python -m finetuning.build_mosaics --speakers s1 s2 s3 s4 s5
"""
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from .config import (
    MOUTH_ROI_DIR,
    MOSAIC_DIR,
    DEFAULT_SPEAKERS,
    MOSAIC_GRID_ROWS,
    MOSAIC_GRID_COLS,
    MOSAIC_CELL_WIDTH,
    MOSAIC_CELL_HEIGHT,
    N_SAMPLE_FRAMES,
)
from .parse_alignments import parse_all_alignments

logger = logging.getLogger(__name__)


def sample_frame_indices(
    speech_start: int,
    speech_end: int,
    n_frames: int = N_SAMPLE_FRAMES,
) -> list[int]:
    """
    Sample n_frames evenly-spaced indices from the speech region.
    Formula: indices[i] = start + i * (end - start) // (n_frames - 1)
    """
    if speech_end <= speech_start:
        speech_end = speech_start + n_frames

    if n_frames == 1:
        return [(speech_start + speech_end) // 2]

    return [
        speech_start + i * (speech_end - speech_start) // (n_frames - 1)
        for i in range(n_frames)
    ]


def build_mosaic(
    mouth_roi_dir: Path,
    frame_indices: list[int],
    rows: int = MOSAIC_GRID_ROWS,
    cols: int = MOSAIC_GRID_COLS,
    cell_w: int = MOSAIC_CELL_WIDTH,
    cell_h: int = MOSAIC_CELL_HEIGHT,
) -> np.ndarray | None:
    """
    Load mouth ROI frames at given indices and arrange into a grid mosaic.
    Returns the mosaic image (BGR) or None if not enough frames found.
    """
    mosaic_w = cols * cell_w
    mosaic_h = rows * cell_h
    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

    loaded = 0
    for i, frame_idx in enumerate(frame_indices):
        frame_path = mouth_roi_dir / f"frame_{frame_idx:03d}.png"
        if not frame_path.exists():
            # Try adjacent frames if exact frame missing
            found = False
            for offset in [1, -1, 2, -2]:
                alt_path = mouth_roi_dir / f"frame_{frame_idx + offset:03d}.png"
                if alt_path.exists():
                    frame_path = alt_path
                    found = True
                    break
            if not found:
                continue

        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        resized = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
        row, col = divmod(i, cols)
        y1 = row * cell_h
        x1 = col * cell_w
        mosaic[y1 : y1 + cell_h, x1 : x1 + cell_w] = resized
        loaded += 1

    # Require at least 75% of frames
    if loaded < len(frame_indices) * 0.75:
        return None

    return mosaic


def build_all_mosaics(
    speakers: list[str],
    mouth_roi_root: Path = MOUTH_ROI_DIR,
    mosaic_root: Path = MOSAIC_DIR,
    limit: int | None = None,
) -> dict[str, bool]:
    """
    Build mosaics for all videos of specified speakers.
    Requires alignments and mouth ROIs to already exist.
    Returns dict of {speaker/video_id: success}.
    """
    # Parse all alignments to get speech boundaries
    alignments = parse_all_alignments(speakers)
    logger.info(f"Loaded {len(alignments)} alignments")

    results: dict[str, bool] = {}

    for speaker in speakers:
        mosaic_speaker_dir = mosaic_root / speaker
        mosaic_speaker_dir.mkdir(parents=True, exist_ok=True)

        speaker_keys = [
            k for k in alignments if k.startswith(f"{speaker}/")
        ]
        if limit is not None:
            speaker_keys = speaker_keys[:limit]

        logger.info(f"Building mosaics for {len(speaker_keys)} videos ({speaker})...")

        built = 0
        skipped = 0
        for idx, key in enumerate(speaker_keys):
            video_id = key.split("/")[1]
            mosaic_path = mosaic_speaker_dir / f"{video_id}.jpg"

            # Skip if already built
            if mosaic_path.exists():
                results[key] = True
                built += 1
                continue

            roi_dir = mouth_roi_root / speaker / video_id
            if not roi_dir.is_dir():
                results[key] = False
                skipped += 1
                continue

            info = alignments[key]
            indices = sample_frame_indices(
                info.speech_start_frame, info.speech_end_frame
            )

            mosaic = build_mosaic(roi_dir, indices)
            if mosaic is None:
                results[key] = False
                skipped += 1
                continue

            cv2.imwrite(
                str(mosaic_path),
                mosaic,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            results[key] = True
            built += 1

            if (idx + 1) % 100 == 0 or (idx + 1) == len(speaker_keys):
                logger.info(
                    f"  {speaker}: {idx+1}/{len(speaker_keys)} "
                    f"(built={built}, skipped={skipped})"
                )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build lip-reading mosaics from mouth ROIs"
    )
    parser.add_argument("--speakers", nargs="+", default=DEFAULT_SPEAKERS)
    parser.add_argument("--mouth-roi-dir", type=Path, default=MOUTH_ROI_DIR)
    parser.add_argument("--mosaic-dir", type=Path, default=MOSAIC_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos per speaker",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    results = build_all_mosaics(
        speakers=args.speakers,
        mouth_roi_root=args.mouth_roi_dir,
        mosaic_root=args.mosaic_dir,
        limit=args.limit,
    )

    success = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Built {success}/{total} mosaics")


if __name__ == "__main__":
    main()
