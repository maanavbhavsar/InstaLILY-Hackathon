"""
Voice output via pyttsx3: speak transcribed phrase at 150 WPM, default audio (headphones).
"""
import pyttsx3

# 150 words per minute (pyttsx3 rate is often in WPM depending on backend)
TARGET_WPM = 150


def get_engine():
    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", TARGET_WPM)
    except Exception:
        pass
    return engine


def speak(phrase: str, engine=None) -> None:
    """Speak the phrase through default audio output (e.g. headphones)."""
    if not phrase or phrase.strip().lower() == "unknown":
        return
    close = False
    if engine is None:
        engine = get_engine()
        close = True
    try:
        engine.say(phrase.strip())
        engine.runAndWait()
    finally:
        if close:
            try:
                engine.stop()
            except Exception:
                pass
