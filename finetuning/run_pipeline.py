from __future__ import annotations

"""
Orchestrate the full GRID lip-reading data preprocessing pipeline.

Steps:
  1. download  — Download GRID corpus from Zenodo
  2. frames    — Extract video frames
  3. rois      — Extract mouth ROIs with MediaPipe
  4. mosaics   — Build 3x4 mosaic images
  5. format    — Create JSONL dataset (train/val/test)
  6. validate  — Run sanity checks

Usage:
    python -m finetuning.run_pipeline --speakers s1
    python -m finetuning.run_pipeline --speakers s1 s2 s3 s4 s5
    python -m finetuning.run_pipeline --speakers s1 --start-from mosaics
    python -m finetuning.run_pipeline --speakers s1 --skip-download
    python -m finetuning.run_pipeline --speakers s1 --limit 10
"""
import argparse
import logging
import time
from pathlib import Path

from .config import DEFAULT_SPEAKERS

logger = logging.getLogger(__name__)

STEPS = ["download", "frames", "rois", "mosaics", "format", "validate"]


def run_pipeline(
    speakers: list[str],
    start_from: str = "download",
    skip_download: bool = False,
    limit: int | None = None,
    workers: int = 4,
) -> bool:
    """
    Run the full preprocessing pipeline.
    Returns True if all steps succeed.
    """
    start_idx = STEPS.index(start_from) if start_from in STEPS else 0

    active_steps = STEPS[start_idx:]
    if skip_download and "download" in active_steps:
        active_steps.remove("download")

    logger.info(f"Pipeline: speakers={speakers}, steps={active_steps}")

    total_start = time.time()

    for step in active_steps:
        step_start = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"  STEP: {step.upper()}", flush=True)
        print(f"{'='*60}", flush=True)

        try:
            if step == "download":
                from .download_grid import download_alignments, download_speaker_videos

                download_alignments()
                for speaker in speakers:
                    download_speaker_videos(speaker)

            elif step == "frames":
                from .extract_frames import extract_all_frames

                results = extract_all_frames(
                    speakers=speakers,
                    max_workers=workers,
                    limit=limit,
                )
                total_frames = sum(results.values())
                print(f"  Extracted {total_frames} frames from {len(results)} videos")

            elif step == "rois":
                from .extract_mouth_rois import extract_all_mouth_rois

                results = extract_all_mouth_rois(
                    speakers=speakers,
                    limit=limit,
                )
                total_success = sum(s for s, _ in results.values())
                total_total = sum(t for _, t in results.values())
                rate = total_success / total_total * 100 if total_total > 0 else 0
                print(
                    f"  Extracted {total_success}/{total_total} mouth ROIs "
                    f"({rate:.1f}% detection rate)"
                )

            elif step == "mosaics":
                from .build_mosaics import build_all_mosaics

                results = build_all_mosaics(
                    speakers=speakers,
                    limit=limit,
                )
                success = sum(1 for v in results.values() if v)
                print(f"  Built {success}/{len(results)} mosaics")

            elif step == "format":
                from .format_dataset import format_dataset

                n_train, n_val, n_test = format_dataset(speakers=speakers)
                print(f"  Dataset: train={n_train}, val={n_val}, test={n_test}")

            elif step == "validate":
                from .validate_dataset import validate_dataset

                ok = validate_dataset(check_images=True)
                if not ok:
                    print("  Validation FAILED")
                    return False

            elapsed = time.time() - step_start
            print(f"  Completed in {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"Step '{step}' failed: {e}", exc_info=True)
            return False

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE ({total_elapsed:.1f}s total)")
    print(f"{'='*60}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run full GRID lip-reading preprocessing pipeline"
    )
    parser.add_argument("--speakers", nargs="+", default=DEFAULT_SPEAKERS)
    parser.add_argument(
        "--start-from",
        choices=STEPS,
        default="download",
        help="Start pipeline from this step",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download step (use existing data)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of videos per speaker (for testing)",
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    ok = run_pipeline(
        speakers=args.speakers,
        start_from=args.start_from,
        skip_download=args.skip_download,
        limit=args.limit,
        workers=args.workers,
    )
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
