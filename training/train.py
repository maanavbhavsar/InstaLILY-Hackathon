from __future__ import annotations

"""
Fine-tune Gemma 3n E4B with LoRA on the GRID lip-reading dataset.

Loads the model in bf16 (full precision, 96GB VRAM), applies LoRA to
attention projection layers, and trains with SFTTrainer from TRL.

Usage:
    # Smoke test (verify pipeline works)
    python3 training/train.py --max-steps 2 --batch-size 1

    # Overfit test (memorize 10 samples)
    python3 training/train.py --num-epochs 50 --batch-size 2 --max-samples 10

    # Full training (~1-2 hours on RTX 6000)
    python3 training/train.py
"""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetuning.config import (
    BASE_MODEL_ID,
    LORA_R,
    LORA_ALPHA,
    LORA_TARGET_MODULES,
    LORA_DROPOUT,
    NUM_EPOCHS,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_STEPS,
    EVAL_STEPS,
    SAVE_STEPS,
    DATASET_DIR,
    ADAPTER_DIR,
)

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 3n for lip reading")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=ADAPTER_DIR.parent)
    parser.add_argument("--model-id", type=str, default=BASE_MODEL_ID)
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=LORA_R)
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Delayed imports so argparse --help works without GPU
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoProcessor
    from trl import SFTConfig, SFTTrainer

    from training.dataset import load_dataset

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Training requires a GPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    logger.info(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

    # Load model
    logger.info(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    # Apply LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    logger.info("Loading datasets...")
    train_data = load_dataset(
        args.dataset_dir, split="train", max_samples=args.max_samples
    )
    val_data = load_dataset(
        args.dataset_dir, split="val", max_samples=args.max_samples
    )

    logger.info(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples")

    # Training config
    output_dir = args.output_dir / "checkpoints"
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=LOGGING_STEPS,
        eval_strategy="steps" if val_data else "no",
        eval_steps=EVAL_STEPS if val_data else None,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        gradient_checkpointing=True,
        max_steps=args.max_steps,
        report_to="none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Collation function for multimodal data
    def collate_fn(examples):
        texts = []
        images = []
        for example in examples:
            msgs = example["messages"]
            # Apply chat template
            text = processor.apply_chat_template(
                msgs, add_generation_prompt=False, tokenize=False
            )
            texts.append(text)
            # Extract image from user message
            for item in msgs[0]["content"]:
                if isinstance(item, dict) and item.get("type") == "image":
                    images.append(item["image"])

        batch = processor(
            text=texts,
            images=images if images else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Set labels = input_ids for causal LM training
        labels = batch["input_ids"].clone()
        # Mask padding tokens in labels
        if processor.tokenizer.pad_token_id is not None:
            labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data if val_data else None,
        data_collator=collate_fn,
        processing_class=processor.tokenizer,
    )

    logger.info("Starting training...")
    trainer.train()

    # Save final adapter
    final_adapter_dir = args.output_dir / "final-adapter"
    logger.info(f"Saving adapter to {final_adapter_dir}")
    model.save_pretrained(str(final_adapter_dir))
    processor.save_pretrained(str(final_adapter_dir))

    logger.info("Training complete!")
    print(f"\nAdapter saved to: {final_adapter_dir}")
    print(f"To evaluate: python3 training/evaluate.py --adapter-dir {final_adapter_dir}")


if __name__ == "__main__":
    main()
