"""
Environmental context for the agent: time, shift, zone, sensors, etc.
Agent uses phrase + this context for LLM-based reasoning before autonomous execution.
"""
from datetime import datetime
from typing import Any


def get_environment_context() -> dict[str, Any]:
    """
    Return current environmental context for agent reasoning.
    In production: wire to real sensors, CMMS, and presence systems.
    """
    now = datetime.now()
    hour = now.hour
    if 6 <= hour < 14:
        shift = "day"
    elif 14 <= hour < 22:
        shift = "swing"
    else:
        shift = "night"
    # Simulated/placeholder fields for demo; replace with real data in production
    return {
        "timestamp": now.isoformat(),
        "shift": shift,
        "zone": "field",
        "temperature": 22,  # °C; from sensor
        "last_maintenance": 14,  # days ago
        "active_tickets": 2,
        "nearby_workers": 3,
    }
