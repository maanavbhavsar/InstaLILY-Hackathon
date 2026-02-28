"""
Agent: LLM or fine-tuned MLP reasons over phrase + context, then we execute the chosen tools.
Decision timing (sub-100ms fine-tuned vs 8–12s base LLM) is the fine-tuning story.
"""
import time
from datetime import datetime
from typing import Any

from .context import get_environment_context
from .reasoning import agent_reason


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def trigger_alert(context: dict[str, Any]) -> dict[str, Any]:
    """Autonomous: raise safety alert (e.g. evacuate, urgent)."""
    msg = "Safety alert raised."
    print(f"[{_ts()}] ACTION: trigger_alert() — {msg}")
    return {"action": "trigger_alert", "status": "executed", "message": msg, "timestamp": _ts()}


def create_ticket(context: dict[str, Any]) -> dict[str, Any]:
    """Autonomous: create maintenance ticket for unit down."""
    msg = "Maintenance ticket created for unit down."
    print(f"[{_ts()}] ACTION: create_ticket() — {msg}")
    return {"action": "create_ticket", "status": "executed", "message": msg, "timestamp": _ts()}


def query_inventory(context: dict[str, Any]) -> dict[str, Any]:
    """Autonomous: query parts inventory."""
    msg = "Inventory lookup requested."
    print(f"[{_ts()}] ACTION: query_inventory() — {msg}")
    return {"action": "query_inventory", "status": "executed", "message": msg, "timestamp": _ts()}


def log_confirmation(context: dict[str, Any]) -> dict[str, Any]:
    """Autonomous: log confirmation to audit."""
    msg = "Confirmation logged."
    print(f"[{_ts()}] ACTION: log_confirmation() — {msg}")
    return {"action": "log_confirmation", "status": "executed", "message": msg, "timestamp": _ts()}


_TOOL_MAP = {
    "trigger_alert": trigger_alert,
    "create_ticket": create_ticket,
    "query_inventory": query_inventory,
    "log_confirmation": log_confirmation,
}


def run_agent(phrase: str, context: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Agent reasons with LLM (Gemma 3) over phrase + context, then we execute the chosen actions.
    Returns (list of execution result dicts, reasoning info {priority, reasoning} for UI).
    """
    context = context or get_environment_context()
    # Time the decision step: fine-tuned MLP <100ms, base Gemma 3 typically 8–12s
    t0 = time.perf_counter()
    result = agent_reason(phrase, context)
    decision_ms = (time.perf_counter() - t0) * 1000
    actions_to_run = result.get("actions") or []
    executions: list[dict[str, Any]] = []
    for action_name in actions_to_run:
        fn = _TOOL_MAP.get(action_name)
        if fn:
            executions.append(fn(context))
    reasoning_info = {
        "priority": result.get("priority", "medium"),
        "reasoning": result.get("reasoning", ""),
        "decision_ms": decision_ms,
        "source": result.get("source", "llm"),
    }
    return executions, reasoning_info
