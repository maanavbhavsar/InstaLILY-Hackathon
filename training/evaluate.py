from __future__ import annotations

"""
Evaluate a fine-tuned Gemma 3n lip-reading adapter.

Computes:
- Word Error Rate (WER)
- Sentence accuracy (exact match)
- Per-position accuracy (command, color, preposition, letter, digit, adverb)

Usage:
    # Full evaluation on test set
    python3 training/evaluate.py --adapter-dir output/final-adapter

    # Quick eval on 10 samples
    python3 training/evaluate.py --adapter-dir output/final-adapter --max-samples 10

    # Test 4-bit quantized inference (simulates laptop deployment)
    python3 training/evaluate.py --adapter-dir output/final-adapter --quantize-4bit
"""
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetuning.config import (
    BASE_MODEL_ID,
    DATASET_DIR,
    ADAPTER_DIR,
    LIP_READING_PROMPT,
)

logger = logging.getLogger(__name__)

POSITION_NAMES = ["command", "color", "preposition", "letter", "digit", "adverb"]


def load_test_samples(dataset_dir: Path, max_samples: int | None = None) -> list[dict]:
    """Load test samples from JSONL, returning (image_path, transcription) pairs."""
    path = dataset_dir / "grid_dataset_test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found: {path}")

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            image_path = data["messages"][0]["content"][0]["image"]
            transcription = data["messages"][1]["content"]
            samples.append({"image_path": image_path, "transcription": transcription})

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples


def evaluate(
    adapter_dir: Path,
    dataset_dir: Path = DATASET_DIR,
    model_id: str = BASE_MODEL_ID,
    max_samples: int | None = None,
    quantize_4bit: bool = False,
) -> dict:
    """
    Run evaluation and return metrics.
    """
    import torch
    from jiwer import wer as compute_wer
    from peft import PeftModel
    from PIL import Image
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    # Load model
    logger.info(f"Loading model: {model_id}")
    load_kwargs = {
        "device_map": "auto",
    }
    if quantize_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        logger.info("Using 4-bit quantization")
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(model_id)

    # Load adapter
    logger.info(f"Loading adapter: {adapter_dir}")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()

    # Load test data
    samples = load_test_samples(dataset_dir, max_samples=max_samples)
    logger.info(f"Evaluating {len(samples)} test samples...")

    predictions = []
    references = []
    position_correct = [0] * 6
    position_total = [0] * 6
    exact_matches = 0

    for idx, sample in enumerate(samples):
        image = Image.open(sample["image_path"]).convert("RGB")
        reference = sample["transcription"]

        # Build chat input
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

        # Decode only the generated tokens
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        prediction = processor.decode(generated_ids, skip_special_tokens=True).strip()

        predictions.append(prediction)
        references.append(reference)

        # Per-position accuracy
        ref_words = reference.split()
        pred_words = prediction.split()
        if len(ref_words) == 6:
            for i in range(6):
                position_total[i] += 1
                if i < len(pred_words) and pred_words[i] == ref_words[i]:
                    position_correct[i] += 1

        # Exact match
        if prediction == reference:
            exact_matches += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            logger.info(f"  Evaluated {idx+1}/{len(samples)} samples")

        # Show first few predictions
        if idx < 5:
            match = "OK" if prediction == reference else "MISS"
            print(f"  [{match}] ref='{reference}' pred='{prediction}'")

    # Compute WER
    word_error_rate = compute_wer(references, predictions)

    # Compute sentence accuracy
    sentence_accuracy = exact_matches / len(samples) if samples else 0

    # Per-position accuracy
    pos_acc = {}
    for i, name in enumerate(POSITION_NAMES):
        if position_total[i] > 0:
            pos_acc[name] = position_correct[i] / position_total[i]
        else:
            pos_acc[name] = 0.0

    results = {
        "wer": word_error_rate,
        "sentence_accuracy": sentence_accuracy,
        "per_position_accuracy": pos_acc,
        "total_samples": len(samples),
        "exact_matches": exact_matches,
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate lip-reading adapter")
    parser.add_argument(
        "--adapter-dir", type=Path, default=ADAPTER_DIR,
    )
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--model-id", type=str, default=BASE_MODEL_ID)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--quantize-4bit", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    results = evaluate(
        adapter_dir=args.adapter_dir,
        dataset_dir=args.dataset_dir,
        model_id=args.model_id,
        max_samples=args.max_samples,
        quantize_4bit=args.quantize_4bit,
    )

    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Samples:            {results['total_samples']}")
    print(f"  WER:                {results['wer']:.4f} ({results['wer']*100:.1f}%)")
    print(f"  Sentence Accuracy:  {results['sentence_accuracy']:.4f} ({results['sentence_accuracy']*100:.1f}%)")
    print(f"  Exact Matches:      {results['exact_matches']}/{results['total_samples']}")
    print()
    print("  Per-Position Accuracy:")
    for name, acc in results["per_position_accuracy"].items():
        print(f"    {name:15s} {acc:.4f} ({acc*100:.1f}%)")
    print(f"{'='*60}")

    # Save results
    results_path = args.adapter_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
