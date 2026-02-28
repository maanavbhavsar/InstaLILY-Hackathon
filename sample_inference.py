from __future__ import annotations

"""
End-to-end demo inference for the lip-reading model.

Supports two modes:
  1. Video mode: raw .mpg video → frames → mouth ROIs → mosaic → model → transcription
  2. Image mode: pre-built mosaic .jpg → model → transcription

Also supports batch mode on the test set.

Usage:
    # From a video file (full pipeline)
    python3 sample_inference.py --video data/grid/raw/s1/bbaf2n.mpg \
        --align data/grid/alignments/s1/bbaf2n.align \
        --adapter-dir models/gemma3n-lipreader-lora/final-adapter

    # From a pre-built mosaic
    python3 sample_inference.py --mosaic data/grid/mosaics/s1/bbaf2n.jpg \
        --adapter-dir models/gemma3n-lipreader-lora/final-adapter

    # 4-bit quantized
    python3 sample_inference.py --mosaic data/grid/mosaics/s1/bbaf2n.jpg \
        --adapter-dir models/gemma3n-lipreader-lora/final-adapter --quantize-4bit

    # Batch mode on test set
    python3 sample_inference.py --test-set data/ \
        --adapter-dir models/gemma3n-lipreader-lora/final-adapter --max-samples 10
"""
import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from finetuning.config import (
    ADAPTER_DIR,
    BASE_MODEL_ID,
    GRID_SAMPLES_PER_FRAME,
    GRID_VIDEO_TOTAL_FRAMES,
    LIP_READING_PROMPT,
    MOSAIC_CELL_HEIGHT,
    MOSAIC_CELL_WIDTH,
    MOSAIC_GRID_COLS,
    MOSAIC_GRID_ROWS,
    N_SAMPLE_FRAMES,
)
from finetuning.build_mosaics import build_mosaic, sample_frame_indices
from finetuning.extract_mouth_rois import _get_face_mesh, extract_mouth_roi
from finetuning.parse_alignments import parse_alignment_file

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Video → mosaic preprocessing
# ---------------------------------------------------------------------------

def video_to_mosaic(
    video_path: Path,
    align_path: Path | None = None,
) -> np.ndarray:
    """
    Extract frames from a video, detect mouth ROIs, and build a mosaic.
    If align_path is provided, uses speech boundaries to select frames.
    Otherwise samples uniformly across the full video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from: {video_path}")

    total_frames = len(frames)
    logger.info(f"Read {total_frames} frames from {video_path.name}")

    # Determine speech boundaries
    if align_path and align_path.exists():
        info = parse_alignment_file(align_path)
        speech_start = info.speech_start_frame
        speech_end = min(info.speech_end_frame, total_frames - 1)
        logger.info(
            f"Alignment: '{info.transcription}' "
            f"(frames {speech_start}-{speech_end})"
        )
    else:
        speech_start = 0
        speech_end = total_frames - 1
        if align_path:
            logger.warning(f"Alignment file not found: {align_path}, using full video")

    # Sample frame indices
    indices = sample_frame_indices(speech_start, speech_end)
    logger.info(f"Sampled frame indices: {indices}")

    # Extract mouth ROIs into a temp directory
    face_mesh = _get_face_mesh()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        detected = 0
        for idx in indices:
            # Clamp index to valid range
            idx = min(idx, total_frames - 1)
            roi = extract_mouth_roi(frames[idx], face_mesh)
            if roi is not None:
                cv2.imwrite(str(tmp_path / f"frame_{idx:03d}.png"), roi)
                detected += 1

        face_mesh.close()
        logger.info(f"Detected mouth in {detected}/{len(indices)} frames")

        if detected < len(indices) * 0.75:
            raise RuntimeError(
                f"Too few mouth detections: {detected}/{len(indices)}"
            )

        mosaic = build_mosaic(tmp_path, indices)

    if mosaic is None:
        raise RuntimeError("Failed to build mosaic from detected ROIs")

    return mosaic


# ---------------------------------------------------------------------------
# Model loading and inference
# ---------------------------------------------------------------------------

def load_model(adapter_dir: Path, model_id: str, quantize_4bit: bool):
    """Load base model + LoRA adapter and processor."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    logger.info(f"Loading base model: {model_id}")
    load_kwargs = {"device_map": "auto"}
    if quantize_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        logger.info("Using 4-bit quantization")
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)

    logger.info(f"Loading adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    return model, processor


def run_inference(model, processor, image) -> str:
    """Run inference on a single mosaic image. Returns predicted transcription."""
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": LIP_READING_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
        )

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return prediction


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Demo inference for the lip-reading model"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--video", type=Path,
        help="Path to a raw .mpg video file (runs full pipeline)",
    )
    group.add_argument(
        "--mosaic", type=Path,
        help="Path to a pre-built mosaic .jpg image",
    )
    group.add_argument(
        "--test-set", type=Path,
        help="Path to dataset dir containing grid_dataset_test.jsonl",
    )

    parser.add_argument(
        "--align", type=Path, default=None,
        help="Path to .align file (optional, used with --video)",
    )
    parser.add_argument(
        "--adapter-dir", type=Path, default=ADAPTER_DIR / "final-adapter",
        help="Path to LoRA adapter directory",
    )
    parser.add_argument(
        "--model-id", type=str, default=BASE_MODEL_ID,
    )
    parser.add_argument(
        "--quantize-4bit", action="store_true",
        help="Use 4-bit quantization for lower memory usage",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples to run in test-set mode",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    from PIL import Image

    # --- Prepare mosaic image(s) ---
    if args.video:
        mosaic_bgr = video_to_mosaic(args.video, args.align)
        mosaic_rgb = cv2.cvtColor(mosaic_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(mosaic_rgb)

        model, processor = load_model(args.adapter_dir, args.model_id, args.quantize_4bit)
        prediction = run_inference(model, processor, image)

        # Show ground truth if alignment available
        gt = ""
        if args.align and args.align.exists():
            info = parse_alignment_file(args.align)
            gt = info.transcription

        print(f"\n{'='*50}")
        print(f"  Video:      {args.video}")
        if gt:
            print(f"  Ground truth: {gt}")
        print(f"  Prediction:   {prediction}")
        if gt:
            match = "MATCH" if prediction == gt else "MISMATCH"
            print(f"  Result:       {match}")
        print(f"{'='*50}")

    elif args.mosaic:
        if not args.mosaic.exists():
            print(f"Error: mosaic not found: {args.mosaic}", file=sys.stderr)
            sys.exit(1)

        image = Image.open(args.mosaic).convert("RGB")

        model, processor = load_model(args.adapter_dir, args.model_id, args.quantize_4bit)
        prediction = run_inference(model, processor, image)

        print(f"\n{'='*50}")
        print(f"  Mosaic:     {args.mosaic}")
        print(f"  Prediction: {prediction}")
        print(f"{'='*50}")

    elif args.test_set:
        jsonl_path = args.test_set / "grid_dataset_test.jsonl"
        if not jsonl_path.exists():
            print(f"Error: test set not found: {jsonl_path}", file=sys.stderr)
            sys.exit(1)

        samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                image_path = data["messages"][0]["content"][0]["image"]
                transcription = data["messages"][1]["content"]
                samples.append({"image_path": image_path, "transcription": transcription})

        if args.max_samples is not None:
            samples = samples[:args.max_samples]

        print(f"Running inference on {len(samples)} test samples...")

        model, processor = load_model(args.adapter_dir, args.model_id, args.quantize_4bit)

        correct = 0
        for idx, sample in enumerate(samples):
            image = Image.open(sample["image_path"]).convert("RGB")
            prediction = run_inference(model, processor, image)
            reference = sample["transcription"]
            match = prediction == reference
            if match:
                correct += 1
            tag = "OK" if match else "MISS"
            print(f"  [{tag}] ref='{reference}' pred='{prediction}'")

        print(f"\n{'='*50}")
        print(f"  Accuracy: {correct}/{len(samples)} ({correct/len(samples)*100:.1f}%)")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
