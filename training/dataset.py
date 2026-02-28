from __future__ import annotations

"""
Dataset loader for GRID lip-reading fine-tuning with SFTTrainer.

Loads JSONL files produced by finetuning/format_dataset.py and converts
them into the chat format expected by Gemma 3n's processor.
"""
import json
import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def load_jsonl(path: Path) -> list[dict]:
    """Load samples from a JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_dataset(
    dataset_dir: Path,
    split: str = "train",
    max_samples: int | None = None,
) -> list[dict]:
    """
    Load a split of the GRID dataset.

    Returns list of dicts with:
      - "messages": chat messages with image loaded as PIL Image
    """
    path = Path(dataset_dir) / f"grid_dataset_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    raw_samples = load_jsonl(path)
    if max_samples is not None:
        raw_samples = raw_samples[:max_samples]

    processed = []
    for sample in raw_samples:
        try:
            msgs = sample["messages"]

            # Find user and assistant messages (may have system message first)
            system_msg = None
            user_msg = None
            assistant_msg = None
            for msg in msgs:
                if msg["role"] == "system":
                    system_msg = msg
                elif msg["role"] == "user":
                    user_msg = msg
                elif msg["role"] == "assistant":
                    assistant_msg = msg

            if user_msg is None or assistant_msg is None:
                continue

            user_content = user_msg["content"]

            # Load image
            image_path = user_content[0]["image"]
            image = Image.open(image_path).convert("RGB")
            prompt_text = user_content[1]["text"]
            transcription = assistant_msg["content"]

            out_msgs = []
            if system_msg is not None:
                out_msgs.append(system_msg)
            out_msgs.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            })
            out_msgs.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": transcription},
                ],
            })

            processed.append({"messages": out_msgs})
        except Exception as e:
            logger.warning(f"Failed to load sample: {e}")
            continue

    logger.info(f"Loaded {len(processed)}/{len(raw_samples)} {split} samples")
    return processed
