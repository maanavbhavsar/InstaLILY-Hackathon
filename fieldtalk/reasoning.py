"""
LLM-based agent reasoning: Gemma 3:4b reasons over phrase + context and decides
which actions to take. This is what separates a rule-based system from a genuine agent.

Priority order:
  1. Base Gemma 3:4b via Ollama — real LLM reasoning with structured output
  2. Hardcoded fallback — simple phrase→action mapping when LLM is unavailable
"""
import os
import re
import threading

os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")
import ollama
from typing import Any

# Model for decision layer: base Gemma 3:4b via Ollama (local, no cloud).
AGENT_MODEL = os.getenv("AGENT_MODEL", "gemma3:4b")
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "120"))

# Known tool names the LLM can choose (including notify_workers from 200-example training).
KNOWN_ACTIONS = {"trigger_alert", "create_ticket", "query_inventory", "log_confirmation", "notify_workers"}
ACTIONS_LIST = ["trigger_alert", "create_ticket", "query_inventory", "log_confirmation"]


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
    Gemma 3:4b reasons over phrase + context and returns priority, actions, reasoning.
    Falls back to hardcoded phrase→action mapping if the LLM is unavailable or times out.
    """
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
                "reasoning": "LLM timeout; using hardcoded fallback.",
                "raw": "",
                "source": "fallback",
            }
        text = (response.get("message") or {}).get("content", "") or ""
        out = _parse_reasoning_response(text, phrase)
        out["source"] = "llm"
        return out
    except Exception as e:
        return {
            "priority": "medium",
            "actions": _fallback_actions(phrase),
            "reasoning": f"LLM unavailable ({e}); using hardcoded fallback.",
            "raw": "",
            "source": "fallback",
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
