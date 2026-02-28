"""
Inference pipeline: local Ollama only (no cloud). Vision model on 16 mouth frames,
returns phrase + confidence. All inference on-device.
"""
import os
import re

# Ensure Ollama client uses localhost only (offline)
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")
import tempfile
from pathlib import Path

import ollama
from .mouth_detection import build_frame_buffer
from .phrase_mapper import map_to_industrial

# Gemma only. Official Ollama vision model is gemma3 (4b/12b/27b have vision). Set MODEL env to override.
MODEL = os.getenv("MODEL", "gemma3:4b")

PHRASES = [
    "stop", "start", "help", "urgent", "clear", "confirmed", "negative",
    "hold", "proceed", "done", "unit down", "need part", "all clear",
    "stand by", "roger", "repeat", "abort", "ready", "check", "evacuate",
]
PHRASE_STR = ", ".join(PHRASES)

PROMPT = (
    "What phrase is this person saying? Choose exactly one from: "
    f"{PHRASE_STR}. "
    "Reply with only the phrase in lowercase, then a space and a number from 0 to 100 for confidence (e.g. 'urgent 85')."
)


def _frames_to_image(mouth_frames: list) -> bytes | None:
    """Build 4x4 grid from 16 mouth frames and return PNG bytes for Ollama."""
    grid = build_frame_buffer(mouth_frames, size=16)
    if grid is None or len(mouth_frames) < 1:
        return None
    import cv2
    ok, buf = cv2.imencode(".png", grid)
    if not ok:
        return None
    return buf.tobytes()


def infer_phrase(mouth_frames: list, model: str | None = None) -> dict:
    """
    Run local inference via Ollama on mouth frame sequence (up to 16 frames as grid).
    Returns {"grid_phrase": str, "industrial_phrase": str, "confidence": float}.
    GRID-style output is mapped to industrial vocabulary via map_to_industrial.
    """
    model = model or MODEL
    image_bytes = _frames_to_image(mouth_frames)
    if image_bytes is None:
        return {"grid_phrase": "unknown", "industrial_phrase": "check", "confidence": 0.0}

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_bytes)
        img_path = f.name
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": PROMPT, "images": [img_path]},
            ],
        )
    finally:
        Path(img_path).unlink(missing_ok=True)

    text = (response.get("message") or {}).get("content", "") or ""
    grid_phrase, confidence = _parse_response(text)
    industrial_phrase = map_to_industrial(grid_phrase)
    return {"grid_phrase": grid_phrase, "industrial_phrase": industrial_phrase, "confidence": confidence}


def _parse_response(text: str) -> tuple[str, float]:
    """Parse model output to raw (GRID-style) phrase and confidence (0-100 -> 0.0-1.0)."""
    text = text.strip().lower()
    match = re.search(r"(\d{1,3})\s*%?", text)
    confidence = 0.5
    if match:
        confidence = min(100, max(0, int(match.group(1)))) / 100.0
        text = text[: match.start()].strip()
    # Raw phrase is whatever the model said (may be GRID-style or industrial)
    grid_phrase = text if text else "unknown"
    return (grid_phrase, confidence)
