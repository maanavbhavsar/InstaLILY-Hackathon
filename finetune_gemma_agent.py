"""
Fine-tune Gemma 3 for FieldTalk agent with Unsloth.
Run: python3 finetune_gemma_agent.py
Uses bf16 (not fp16) for Gemma 3 compatibility with Unsloth.
"""
import json
import os
from pathlib import Path

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Load data: FIELDTALK_AGENT_DATA env (e.g. industrial_safety_200.jsonl) or default synthetic
SCRIPT_DIR = Path(__file__).resolve().parent
_data_path = os.environ.get("FIELDTALK_AGENT_DATA")
if _data_path:
    DATA_PATH = Path(_data_path)
else:
    # Prefer 200-example industrial set if present
    industrial_200 = SCRIPT_DIR / "industrial_safety_200.jsonl"
    DATA_PATH = industrial_200 if industrial_200.is_file() else (SCRIPT_DIR / "data" / "synthetic_agent_data.jsonl")
with open(DATA_PATH, encoding="utf-8") as f:
    examples = [json.loads(l) for l in f if l.strip()]

print(f"Loaded {len(examples)} examples from {DATA_PATH}")


def format_example(ex):
    ctx = ex.get("context", {})
    # Normalize temperature (200-example data uses strings like "extreme", "normal")
    temp = ctx.get("temperature", "")
    if isinstance(temp, (int, float)):
        temp = str(temp)
    actions = ", ".join(ex["actions"]) if ex.get("actions") else "none"
    return {
        "text": f"""<start_of_turn>user
You are an autonomous industrial safety agent called FieldTalk.
Phrase: {ex['phrase']}
Zone: {ctx.get('zone', '')}
Shift: {ctx.get('shift', '')}
Temperature: {temp}
Last maintenance: {ctx.get('last_maintenance', '')} days
Active tickets: {ctx.get('active_tickets', '')}
Nearby workers: {ctx.get('nearby_workers', '')}
Decide: priority, actions, reasoning.
<end_of_turn>
<start_of_turn>model
Priority: {ex.get('priority', '')}
Actions: {actions}
Reasoning: {ex.get('reasoning', '')}
<end_of_turn>"""
    }


dataset = Dataset.from_list([format_example(e) for e in examples])

model, tokenizer = FastLanguageModel.from_pretrained(
    "google/gemma-3-4b-it",
    max_seq_length=512,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Gemma 3 / Unsloth: use bf16, not fp16
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    args=TrainingArguments(
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        max_steps=200,
        learning_rate=2e-4,
        fp16=False,
        bf16=True,
        output_dir="fieldtalk_agent_model",
        logging_steps=20,
    ),
)

trainer.train()

# Always save merged 16-bit model (HuggingFace format) so we have a usable checkpoint
# even if GGUF conversion fails (e.g. exit 127 when `python` is not in PATH).
merge_dir = "fieldtalk_agent_model_merged"
model.save_pretrained_merged(merge_dir, tokenizer, save_method="merged_16bit")
print(f"Merged model saved to {merge_dir}/")

# GGUF conversion requires llama.cpp and often `python` in PATH (many systems only have `python3`).
# If it fails, use the merged model above or convert manually.
try:
    model.save_pretrained_gguf(
        "fieldtalk_agent_gguf",
        tokenizer,
        quantization_method="q4_k_m",
    )
    print("GGUF model saved to fieldtalk_agent_gguf/")
except RuntimeError as e:
    if "GGUF" in str(e) or "127" in str(e):
        print(
            "GGUF conversion failed (often due to `python` not found; try `ln -sf $(which python3) /usr/bin/python` or convert manually)."
        )
        print(f"Merged model is in {merge_dir}/ — use that for inference.")
    else:
        raise

print("Training complete.")
