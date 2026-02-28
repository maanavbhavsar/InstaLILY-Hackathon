"""
LLM-based agent reasoning: Gemma 3 reasons over phrase + context and decides
which actions to take. This is what separates a rule-based system from a genuine agent.
"""
import os
import re
import threading

os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")
import ollama
from typing import Any

# Model for decision layer (not lip reading). Gemma 3 for contextual reasoning.
AGENT_MODEL = os.getenv("AGENT_MODEL", "gemma3:4b")
# Timeout (seconds) for Ollama call so app doesn't hang if model is slow/missing
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "30"))

# Known tool names the LLM can choose; we map these to actual functions
KNOWN_ACTIONS = {"trigger_alert", "create_ticket", "query_inventory", "log_confirmation"}
ACTIONS_LIST = ["trigger_alert", "create_ticket", "query_inventory", "log_confirmation"]

# Optional fine-tuned model (script: scripts/finetune_agent.py). Auto-use if path unset but default exists.
_default_finetuned = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "finetuned_agent.pt")
FINE_TUNED_AGENT_PATH = os.getenv("FINE_TUNED_AGENT_PATH", "").strip() or (_default_finetuned if os.path.isfile(_default_finetuned) else "")
_finetuned_model = None
_finetuned_vocabs = None


def _load_finetuned():
    global _finetuned_model, _finetuned_vocabs
    if _finetuned_model is not None or not FINE_TUNED_AGENT_PATH or not os.path.isfile(FINE_TUNED_AGENT_PATH):
        return _finetuned_model is not None
    try:
        import torch
        try:
            ckpt = torch.load(FINE_TUNED_AGENT_PATH, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(FINE_TUNED_AGENT_PATH, map_location="cpu")
    except Exception:
        return False
    try:
        from torch import nn
        class AgentMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(7, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 4), nn.Sigmoid()
                )
            def forward(self, x):
                return self.net(x)
        _finetuned_model = AgentMLP()
        _finetuned_model.load_state_dict(ckpt["model_state"])
        _finetuned_model.eval()
        _finetuned_vocabs = ckpt.get("vocabs")
        return True
    except Exception:
        return False


def _encode_for_finetuned(phrase: str, context: dict[str, Any]) -> list[float]:
    """Encode (phrase, context) as 7-dim vector for the fine-tuned MLP."""
    PHRASES = [
        "stop", "start", "help", "urgent", "clear", "confirmed", "negative",
        "hold", "proceed", "done", "unit down", "need part", "all clear",
        "stand by", "roger", "repeat", "abort", "ready", "check", "evacuate",
    ]
    SHIFTS, ZONES = ["day", "swing", "night"], ["field"]
    phrase_to_id = {p: i for i, p in enumerate(PHRASES)}
    shift_to_id = {s: i for i, s in enumerate(SHIFTS)}
    zone_to_id = {z: i for i, z in enumerate(ZONES)}
    pid = phrase_to_id.get(phrase.strip().lower(), 0)
    sid = shift_to_id.get(context.get("shift", "day"), 0)
    zid = zone_to_id.get(context.get("zone", "field"), 0)
    temp = float(context.get("temperature", 22)) / 30.0
    last_m = float(context.get("last_maintenance", 14)) / 30.0
    tickets = float(context.get("active_tickets", 0)) / 10.0
    workers = float(context.get("nearby_workers", 2)) / 10.0
    return [float(pid), float(sid), float(zid), temp, last_m, tickets, workers]


def _predict_actions_finetuned(phrase: str, context: dict[str, Any]) -> list[str]:
    """Return list of action names from the fine-tuned model if loaded."""
    if not _load_finetuned() or _finetuned_model is None:
        return []
    try:
        import torch
        x = torch.tensor([_encode_for_finetuned(phrase, context)], dtype=torch.float32)
        with torch.no_grad():
            out = _finetuned_model(x)
        out = out[0].tolist()
        return [ACTIONS_LIST[i] for i in range(4) if out[i] >= 0.5]
    except Exception:
        return []


def _build_prompt(phrase: str, context: dict[str, Any]) -> str:
    return f'''
You are an autonomous industrial agent.

Worker mouthed: "{phrase}"
Current environment:
- Zone: {context.get("zone", "—")}
- Shift: {context.get("shift", "—")}
- Temperature: {context.get("temperature", "—")}°C
- Last maintenance: {context.get("last_maintenance", "—")} days ago
- Active tickets: {context.get("active_tickets", "—")}
- Nearby workers: {context.get("nearby_workers", "—")}

Decide what autonomous actions to take and why.
You may choose one or more of: trigger_alert, create_ticket, query_inventory, log_confirmation.
If no action is needed, say NONE.

Reply in this exact format:
PRIORITY: <high|medium|low>
ACTIONS: <comma-separated action names, or NONE>
REASONING: <short explanation>
'''.strip()


def _ollama_chat_with_timeout(model: str, messages: list, timeout_sec: int) -> dict | None:
    result = [None]
    def _run():
        result[0] = ollama.chat(model=model, messages=messages)
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout_sec)
    if t.is_alive():
        return None  # timeout
    return result[0]


def agent_reason(phrase: str, context: dict[str, Any], model: str | None = None) -> dict[str, Any]:
    """
    LLM or fine-tuned model reasons over phrase + context and returns priority, actions, reasoning.
    When FINE_TUNED_AGENT_PATH is set, the fine-tuned component is used for actions.
    """
    # Prefer fine-tuned model when available: sub-100ms, deterministic (safety-critical)
    actions_ft = _predict_actions_finetuned(phrase, context)
    if actions_ft:
        return {
            "priority": "medium",
            "actions": actions_ft,
            "reasoning": "Fine-tuned agent decision (trained on synthetic phrase+context→actions). Sub-100ms for safety-critical execution.",
            "raw": "",
            "source": "finetuned",
        }

    model = model or AGENT_MODEL
    prompt = _build_prompt(phrase, context)
    try:
        response = _ollama_chat_with_timeout(
            model, [{"role": "user", "content": prompt}], AGENT_TIMEOUT
        )
        if response is None:
            return {
                "priority": "medium",
                "actions": _fallback_actions(phrase),
                "reasoning": "LLM timeout; using fallback.",
                "raw": "",
                "source": "llm",
            }
        text = (response.get("message") or {}).get("content", "") or ""
        out = _parse_reasoning_response(text, phrase)
        out["source"] = "llm"
        return out
    except Exception as e:
        return {
            "priority": "medium",
            "actions": _fallback_actions(phrase),
            "reasoning": f"LLM unavailable ({e}); using fallback.",
            "raw": "",
            "source": "llm",
        }


def _parse_reasoning_response(text: str, phrase: str) -> dict[str, Any]:
    """Parse LLM output for PRIORITY, ACTIONS, REASONING."""
    lines = text.strip().split("\n")
    priority = "medium"
    actions: list[str] = []
    reasoning = ""

    for line in lines:
        line = line.strip()
        if line.upper().startswith("PRIORITY:"):
            p = line.split(":", 1)[-1].strip().lower()
            if p in ("high", "medium", "low"):
                priority = p
        elif line.upper().startswith("ACTIONS:"):
            rest = line.split(":", 1)[-1].strip().upper()
            if rest == "NONE" or not rest:
                pass
            else:
                for part in re.split(r"[,;]", rest):
                    a = part.strip().lower().replace(" ", "_")
                    if a in KNOWN_ACTIONS:
                        actions.append(a)
        elif line.upper().startswith("REASONING:"):
            reasoning = line.split(":", 1)[-1].strip()

    # Only fall back to phrase lookup when parse failed (no reasoning); if LLM said NONE, keep actions empty
    if not actions and not reasoning:
        actions = _fallback_actions(phrase)

    return {
        "priority": priority,
        "actions": actions,
        "reasoning": reasoning or "No reasoning provided.",
        "raw": text,
        "source": "llm",
    }


def _fallback_actions(phrase: str) -> list[str]:
    """Fallback when LLM fails or returns no actions: simple phrase-to-action mapping."""
    phrase_lower = phrase.strip().lower()
    if phrase_lower in ("urgent", "evacuate"):
        return ["trigger_alert"]
    if phrase_lower == "unit down":
        return ["create_ticket"]
    if phrase_lower == "need part":
        return ["query_inventory"]
    if phrase_lower == "confirmed":
        return ["log_confirmation"]
    return []
