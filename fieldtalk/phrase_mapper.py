"""
Map GRID Corpus phrases to industrial vocabulary.
Used after lip-reading / vision output to normalize to field commands.
"""

# Industrial phrases we accept (pass-through if already matched)
INDUSTRIAL_PHRASES = {
    "stop", "start", "help", "urgent", "clear", "confirmed", "negative",
    "hold", "proceed", "done", "unit down", "need part", "all clear",
    "stand by", "roger", "repeat", "abort", "ready", "check", "evacuate",
}

GRID_TO_INDUSTRIAL = {
    "bin": "unit down",
    "lay": "need part",
    "place": "confirmed",
    "set": "check",
    "move": "proceed",
    "blue": "urgent",
    "red": "stop",
    "green": "clear",
    "white": "help",
    "now": "urgent",
    "soon": "stand by",
    "please": "confirmed",
    "again": "repeat",
    "no": "negative",
    "yes": "confirmed",
    "go": "start",
    "stop": "stop",
    "in": "hold",
    "by": "stand by",
    "at": "check",
}


def map_to_industrial(grid_phrase: str) -> str:
    """
    Maps GRID corpus output to industrial vocabulary.
    Checks each word in the GRID phrase against mapping.
    Returns matched industrial phrase or "check" as default.
    """
    if not grid_phrase or not isinstance(grid_phrase, str):
        return "check"
    normalized = grid_phrase.strip().lower()
    if normalized in INDUSTRIAL_PHRASES:
        return normalized
    words = normalized.split()
    for word in words:
        if word in GRID_TO_INDUSTRIAL:
            return GRID_TO_INDUSTRIAL[word]
    return "check"
