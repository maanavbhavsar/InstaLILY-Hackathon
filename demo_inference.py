"""
Demo inference script: process a video file and get a GRID lip-reading
transcription from a fine-tuned Gemma 3 4B GGUF served by Ollama.

Reuses the same preprocessing (mouth ROI extraction, 3x4 mosaic) as the
training pipeline so the model sees exactly what it was trained on.

Usage:
    python demo_inference.py --video path/to/demo.mp4
    python demo_inference.py --video path/to/demo.mp4 --model fieldtalk-lipreader
    python demo_inference.py --video demo.mp4 --align demo.align
    python demo_inference.py --video demo.mp4 --save-mosaic mosaic.jpg
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Ensure Ollama client uses localhost only
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")
import ollama

# --- Reuse finetuning pipeline components ---
from finetuning.config import (
    LIP_READING_PROMPT,
    LIP_READING_SYSTEM_PROMPT,
    MOSAIC_CELL_HEIGHT,
    MOSAIC_CELL_WIDTH,
    MOSAIC_GRID_COLS,
    MOSAIC_GRID_ROWS,
    N_SAMPLE_FRAMES,
)
from finetuning.extract_mouth_rois import extract_mouth_roi, _get_face_mesh
from finetuning.build_mosaics import sample_frame_indices
from finetuning.parse_alignments import parse_alignment_file


def extract_frames(video_path: str) -> list[np.ndarray]:
    """Read all frames from a video file. Returns list of BGR frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video '{video_path}'")
        sys.exit(1)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def build_mosaic_from_rois(
    rois: list[np.ndarray],
    rows: int = MOSAIC_GRID_ROWS,
    cols: int = MOSAIC_GRID_COLS,
    cell_w: int = MOSAIC_CELL_WIDTH,
    cell_h: int = MOSAIC_CELL_HEIGHT,
) -> np.ndarray:
    """Arrange mouth ROI images into a rows x cols grid mosaic (in memory)."""
    mosaic = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)
    for i, roi in enumerate(rois):
        if roi is None:
            continue
        resized = cv2.resize(roi, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
        row, col = divmod(i, cols)
        y1 = row * cell_h
        x1 = col * cell_w
        mosaic[y1 : y1 + cell_h, x1 : x1 + cell_w] = resized
    return mosaic


def run_ollama(mosaic: np.ndarray, model: str) -> str:
    """Send the mosaic PNG to Ollama and return the model's response text."""
    ok, buf = cv2.imencode(".png", mosaic)
    if not ok:
        print("Error: failed to encode mosaic as PNG")
        sys.exit(1)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(buf.tobytes())
        img_path = f.name

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": LIP_READING_SYSTEM_PROMPT},
                {"role": "user", "content": LIP_READING_PROMPT, "images": [img_path]},
            ],
        )
    finally:
        Path(img_path).unlink(missing_ok=True)

    return (response.get("message") or {}).get("content", "").strip()


def main():
    parser = argparse.ArgumentParser(
        description="Run lip-reading inference on a demo video via Ollama"
    )
    parser.add_argument(
        "--video", required=True, help="Path to the input video file"
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL", "fieldtalk-lipreader"),
        help="Ollama model name (default: $MODEL or fieldtalk-lipreader)",
    )
    parser.add_argument(
        "--align",
        default=None,
        help="Optional .align file (GRID format) to restrict to speech frames",
    )
    parser.add_argument(
        "--save-mosaic",
        default=None,
        help="Save the mosaic image to this path for inspection",
    )
    args = parser.parse_args()

    # 1. Extract all frames from video
    print(f"Reading video: {args.video}")
    frames = extract_frames(args.video)
    total_frames = len(frames)
    print(f"  Extracted {total_frames} frames")

    if total_frames == 0:
        print("Error: no frames extracted from video")
        sys.exit(1)

    # 2. Determine speech boundaries
    if args.align:
        print(f"Using alignment file: {args.align}")
        align_info = parse_alignment_file(
            Path(args.align), total_frames=total_frames
        )
        speech_start = align_info.speech_start_frame
        speech_end = align_info.speech_end_frame
        print(f"  Speech region: frames {speech_start}-{speech_end}")
        print(f"  Ground truth: \"{align_info.transcription}\"")
    else:
        speech_start = 0
        speech_end = total_frames - 1

    # 3. Sample 12 evenly-spaced frame indices
    indices = sample_frame_indices(speech_start, speech_end, N_SAMPLE_FRAMES)
    # Clamp to valid range
    indices = [min(i, total_frames - 1) for i in indices]
    print(f"  Sampled frame indices: {indices}")

    # 4. Extract mouth ROIs from sampled frames
    print("Extracting mouth ROIs with MediaPipe...")
    face_mesh = _get_face_mesh()
    rois = []
    for idx in indices:
        roi = extract_mouth_roi(frames[idx], face_mesh)
        if roi is None:
            print(f"  Warning: no face detected in frame {idx}")
        rois.append(roi)
    face_mesh.close()

    detected = sum(1 for r in rois if r is not None)
    print(f"  Detected mouths in {detected}/{len(indices)} frames")

    if detected == 0:
        print("Error: no mouths detected in any sampled frame")
        sys.exit(1)

    # 5. Build 3x4 mosaic
    mosaic = build_mosaic_from_rois(rois)

    if args.save_mosaic:
        cv2.imwrite(args.save_mosaic, mosaic, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  Mosaic saved to: {args.save_mosaic}")

    # 6. Send to Ollama
    print(f"Running inference with model '{args.model}'...")
    transcription = run_ollama(mosaic, args.model)

    print()
    print(f"Transcription: {transcription}")

    if args.align:
        print(f"Ground truth:  {align_info.transcription}")


if __name__ == "__main__":
    main()
