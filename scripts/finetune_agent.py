"""
Fine-tune the agent decision layer on synthetic (phrase, context) -> actions.
Produces a small PyTorch model that can be loaded in the app as the fine-tuned component.
Run from project root: python scripts/finetune_agent.py
"""
import json
import os
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "data" / "synthetic_agent_data.jsonl"
OUT_DIR = ROOT / "models"
OUT_PATH = OUT_DIR / "finetuned_agent.pt"

# Vocabulary (must match agent_layer and reasoning)
PHRASES = [
    "stop", "start", "help", "urgent", "clear", "confirmed", "negative",
    "hold", "proceed", "done", "unit down", "need part", "all clear",
    "stand by", "roger", "repeat", "abort", "ready", "check", "evacuate",
]
SHIFTS = ["day", "swing", "night"]
ZONES = ["field"]
ACTIONS = ["trigger_alert", "create_ticket", "query_inventory", "log_confirmation"]


def load_data(path: Path):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def encode(examples):
    """Encode (phrase, context) -> feature vector; actions -> multi-label vector."""
    X = []
    y = []
    phrase_to_id = {p: i for i, p in enumerate(PHRASES)}
    shift_to_id = {s: i for i, s in enumerate(SHIFTS)}
    zone_to_id = {z: i for i, z in enumerate(ZONES)}
    action_to_id = {a: i for i, a in enumerate(ACTIONS)}

    for ex in examples:
        phrase = ex["phrase"].strip().lower()
        ctx = ex.get("context", {})
        pid = phrase_to_id.get(phrase, 0)
        sid = shift_to_id.get(ctx.get("shift", "day"), 0)
        zid = zone_to_id.get(ctx.get("zone", "field"), 0)
        temp = float(ctx.get("temperature", 22)) / 30.0
        last_m = float(ctx.get("last_maintenance", 14)) / 30.0
        tickets = float(ctx.get("active_tickets", 0)) / 10.0
        workers = float(ctx.get("nearby_workers", 2)) / 10.0
        X.append([pid, sid, zid, temp, last_m, tickets, workers])
        labels = [0.0] * len(ACTIONS)
        for a in ex.get("actions", []):
            if a in action_to_id:
                labels[action_to_id[a]] = 1.0
        y.append(labels)
    return X, y, (phrase_to_id, shift_to_id, zone_to_id, action_to_id)


def main():
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch required for fine-tuning. Install: pip install torch")
        sys.exit(1)

    if not DATA_PATH.exists():
        print(f"Data not found: {DATA_PATH}")
        sys.exit(1)

    examples = load_data(DATA_PATH)
    print(f"Loaded {len(examples)} examples from {DATA_PATH}")
    X, y, vocabs = encode(examples)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    # Small MLP: 7 -> 32 -> 16 -> 4 (multi-label)
    class AgentMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(7, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 4),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    model = AgentMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCELoss()
    for epoch in range(200):
        opt.zero_grad()
        out = model(X_t)
        loss = loss_fn(out, y_t)
        loss.backward()
        opt.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1} loss={loss.item():.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "vocabs": vocabs,
        "phrases": PHRASES,
        "actions": ACTIONS,
    }, OUT_PATH)
    print(f"Saved fine-tuned model to {OUT_PATH}")
    print("Set FINE_TUNED_AGENT_PATH to this path in the app to use the fine-tuned component.")


if __name__ == "__main__":
    main()
