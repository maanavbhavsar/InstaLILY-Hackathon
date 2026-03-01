from __future__ import annotations

"""
Extract mouth ROI crops from video frames using MediaPipe Face Mesh.

Uses the same landmark indices and padding as the live app
(fieldtalk/mouth_detection.py) but with static_image_mode=True
for batch processing.

Usage:
    python -m finetuning.extract_mouth_rois --speakers s1
    python -m finetuning.extract_mouth_rois --speakers s1 --limit 1
    python -m finetuning.extract_mouth_rois --speakers s1 s2 s3 s4 s5
"""
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from .config import (
    FRAMES_DIR,
    MOUTH_ROI_DIR,
    DEFAULT_SPEAKERS,
    MEDIAPIPE_MOUTH_INDICES,
    MOUTH_PADDING,
)

logger = logging.getLogger(__name__)


def _get_face_mesh():
    """Create a MediaPipe FaceMesh instance for batch (static) processing."""
    import mediapipe as mp

    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


def extract_mouth_roi(
    frame: np.ndarray,
    face_mesh,
    padding: float = MOUTH_PADDING,
) -> np.ndarray | None:
    """
    Detect face and crop mouth region from a single frame (BGR).
    Uses the same landmark indices and padding logic as the live app.
    Returns the cropped mouth region (BGR) or None if no face detected.
    """
    h_img, w_img = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    xs = []
    ys = []
    for i in MEDIAPIPE_MOUTH_INDICES:
        if i < len(landmarks):
            lm = landmarks[i]
            xs.append(lm.x * w_img)
            ys.append(lm.y * h_img)

    if not xs or not ys:
        return None

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    pad_w = w * padding
    pad_h = h * padding

    x1 = max(0, int(x_min - pad_w))
    y1 = max(0, int(y_min - pad_h))
    x2 = min(w_img, int(x_max + pad_w))
    y2 = min(h_img, int(y_max + pad_h))

    if x2 - x1 < 10 or y2 - y1 < 10:
        return None

    return frame[y1:y2, x1:x2].copy()


def extract_mouth_rois_for_video(
    frames_dir: Path,
    output_dir: Path,
    face_mesh,
    padding: float = MOUTH_PADDING,
) -> tuple[int, int]:
    """
    Extract mouth ROIs from all frames in a video directory.
    Returns (success_count, total_count).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        return 0, 0

    success = 0
    for frame_path in frame_files:
        out_path = output_dir / frame_path.name
        if out_path.exists():
            success += 1
            continue

        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        roi = extract_mouth_roi(frame, face_mesh, padding=padding)
        if roi is not None:
            cv2.imwrite(str(out_path), roi)
            success += 1

    return success, len(frame_files)


def extract_all_mouth_rois(
    speakers: list[str],
    frames_root: Path = FRAMES_DIR,
    output_root: Path = MOUTH_ROI_DIR,
    padding: float = MOUTH_PADDING,
    limit: int | None = None,
) -> dict[str, tuple[int, int]]:
    """
    Extract mouth ROIs for all videos of specified speakers.
    MediaPipe is NOT thread-safe, so this runs sequentially.
    Returns dict of {speaker/video_id: (success_count, total_count)}.
    """
    face_mesh = _get_face_mesh()
    results: dict[str, tuple[int, int]] = {}
    skipped_videos = 0

    for speaker in speakers:
        speaker_frames_dir = frames_root / speaker
        if not speaker_frames_dir.is_dir():
            logger.warning(f"Frames directory not found: {speaker_frames_dir}")
            continue

        video_dirs = sorted(
            [d for d in speaker_frames_dir.iterdir() if d.is_dir()]
        )
        if limit is not None:
            video_dirs = video_dirs[:limit]

        logger.info(f"Processing {len(video_dirs)} videos for {speaker}...")

        for idx, video_dir in enumerate(video_dirs):
            video_id = video_dir.name
            key = f"{speaker}/{video_id}"
            output_dir = output_root / speaker / video_id

            # Skip if already done
            if output_dir.is_dir():
                existing = len(list(output_dir.glob("frame_*.png")))
                total = len(list(video_dir.glob("frame_*.png")))
                if existing >= total * 0.8:
                    results[key] = (existing, total)
                    continue

            success, total = extract_mouth_rois_for_video(
                video_dir, output_dir, face_mesh, padding=padding
            )
            results[key] = (success, total)

            # Warn if too many failures
            if total > 0 and success / total < 0.8:
                logger.warning(
                    f"Low detection rate for {key}: {success}/{total} "
                    f"({success/total*100:.0f}%)"
                )
                skipped_videos += 1

            if (idx + 1) % 50 == 0 or (idx + 1) == len(video_dirs):
                logger.info(
                    f"  {speaker}: {idx+1}/{len(video_dirs)} videos processed"
                )

    face_mesh.close()

    if skipped_videos > 0:
        logger.warning(
            f"{skipped_videos} videos had low face detection rates (<80%)"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract mouth ROIs from GRID video frames"
    )
    parser.add_argument("--speakers", nargs="+", default=DEFAULT_SPEAKERS)
    parser.add_argument("--frames-dir", type=Path, default=FRAMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=MOUTH_ROI_DIR)
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

    results = extract_all_mouth_rois(
        speakers=args.speakers,
        frames_root=args.frames_dir,
        output_root=args.output_dir,
        limit=args.limit,
    )

    total_videos = len(results)
    total_success = sum(s for s, _ in results.values())
    total_frames = sum(t for _, t in results.values())
    rate = total_success / total_frames * 100 if total_frames > 0 else 0
    print(
        f"Processed {total_videos} videos: "
        f"{total_success}/{total_frames} frames ({rate:.1f}% detection rate)"
    )


if __name__ == "__main__":
    main()
