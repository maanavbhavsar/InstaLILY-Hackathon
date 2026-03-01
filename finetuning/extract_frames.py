from __future__ import annotations

"""
Extract individual frames from GRID .mpg video files using OpenCV.

Usage:
    python -m finetuning.extract_frames --speakers s1 s2 s3 s4 s5
    python -m finetuning.extract_frames --speakers s1 --limit 1
    python -m finetuning.extract_frames --speakers s1 --workers 8
"""
import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

from .config import RAW_VIDEO_DIR, FRAMES_DIR, DEFAULT_SPEAKERS

logger = logging.getLogger(__name__)


def extract_frames_from_video(video_path: Path, output_dir: Path) -> int:
    """
    Extract all frames from a single .mpg video.
    Saves as frame_000.png, frame_001.png, etc.
    Returns the number of frames extracted.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out_path = output_dir / f"frame_{idx:03d}.png"
        cv2.imwrite(str(out_path), frame)
        idx += 1

    cap.release()
    return idx


def _extract_worker(args: tuple[Path, Path]) -> tuple[str, int]:
    """Worker function for parallel extraction."""
    video_path, output_dir = args
    count = extract_frames_from_video(video_path, output_dir)
    return video_path.stem, count


def extract_all_frames(
    speakers: list[str],
    raw_dir: Path = RAW_VIDEO_DIR,
    frames_dir: Path = FRAMES_DIR,
    max_workers: int = 4,
    limit: int | None = None,
) -> dict[str, int]:
    """
    Extract frames from all videos for the specified speakers.
    Uses multiprocessing for speed.
    Returns dict of {speaker/video_id: frame_count}.
    """
    results: dict[str, int] = {}
    tasks: list[tuple[Path, Path]] = []

    for speaker in speakers:
        speaker_video_dir = raw_dir / speaker
        if not speaker_video_dir.is_dir():
            logger.warning(f"Video directory not found: {speaker_video_dir}")
            continue

        videos = sorted(speaker_video_dir.glob("*.mpg"))
        if limit is not None:
            videos = videos[:limit]

        for video_path in videos:
            video_id = video_path.stem
            output_dir = frames_dir / speaker / video_id
            # Skip if already extracted
            if output_dir.is_dir() and len(list(output_dir.glob("frame_*.png"))) >= 70:
                results[f"{speaker}/{video_id}"] = len(list(output_dir.glob("frame_*.png")))
                continue
            tasks.append((video_path, output_dir))

    if not tasks:
        logger.info("No new videos to extract (all already done or no videos found)")
        return results

    logger.info(f"Extracting frames from {len(tasks)} videos using {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_extract_worker, t): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            video_path, output_dir = futures[future]
            speaker = video_path.parent.name
            try:
                video_id, count = future.result()
                key = f"{speaker}/{video_id}"
                results[key] = count
                done += 1
                if done % 100 == 0 or done == len(tasks):
                    logger.info(f"  Extracted {done}/{len(tasks)} videos")
            except Exception as e:
                logger.warning(f"Failed to extract {video_path}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract frames from GRID videos")
    parser.add_argument("--speakers", nargs="+", default=DEFAULT_SPEAKERS)
    parser.add_argument("--raw-dir", type=Path, default=RAW_VIDEO_DIR)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of videos per speaker")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    results = extract_all_frames(
        speakers=args.speakers,
        raw_dir=args.raw_dir,
        frames_dir=args.frames_dir,
        max_workers=args.workers,
        limit=args.limit,
    )

    total_videos = len(results)
    total_frames = sum(results.values())
    print(f"Extracted {total_frames} frames from {total_videos} videos")


if __name__ == "__main__":
    main()
