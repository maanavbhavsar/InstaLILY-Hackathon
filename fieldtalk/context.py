"""
Environmental context for the agent: time, shift, zone, sensors, etc.
Agent uses phrase + this context for LLM-based reasoning before autonomous execution.
"""
from datetime import datetime
from typing import Any

# Demo scenarios with predefined context values (for sidebar dropdown)
DEMO_SCENARIOS = {
    "Normal": {
        "shift": "day",
        "zone": "field",
        "temperature": 22,
        "last_maintenance": 7,
        "active_tickets": 0,
        "nearby_workers": 4,
    },
    "High Risk": {
        "shift": "swing",
        "zone": "field",
        "temperature": 35,
        "last_maintenance": 28,
        "active_tickets": 3,
        "nearby_workers": 1,
    },
    "Emergency": {
        "shift": "night",
        "zone": "field",
        "temperature": 42,
        "last_maintenance": 45,
        "active_tickets": 5,
        "nearby_workers": 0,
    },
}


def get_environment_context(demo_scenario: str | None = None) -> dict[str, Any]:
    """
    Return current environmental context for agent reasoning.
    If demo_scenario is set ("Normal", "High Risk", "Emergency"), returns that scenario's context.
    Otherwise uses live time for shift and placeholder values.
    """
    now = datetime.now()
    base = {"timestamp": now.isoformat()}

    if demo_scenario and demo_scenario in DEMO_SCENARIOS:
        base.update(DEMO_SCENARIOS[demo_scenario])
        return base

    hour = now.hour
    if 6 <= hour < 14:
        shift = "day"
    elif 14 <= hour < 22:
        shift = "swing"
    else:
        shift = "night"
    base.update({
        "shift": shift,
        "zone": "field",
        "temperature": 22,
        "last_maintenance": 14,
        "active_tickets": 2,
        "nearby_workers": 3,
    })
    return base
