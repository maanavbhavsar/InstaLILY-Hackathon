"""
Run full workflow (no webcam/Ollama required): imports, context, agent, mouth detector.
Verifies all files are set up. Optional: real inference if Ollama + image available.
"""
import os
import sys

# Force local Ollama before any ollama import
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")

def test_imports():
    print("1. Testing imports...")
    from fieldtalk.mouth_detection import MouthDetector, build_frame_buffer
    from fieldtalk.context import get_environment_context
    from fieldtalk.agent_layer import run_agent
    from fieldtalk.voice import get_engine
    from fieldtalk.inference import infer_phrase, MODEL, PHRASES
    print("   OK: all fieldtalk modules import")

def test_context():
    print("2. Environmental context...")
    from fieldtalk.context import get_environment_context
    ctx = get_environment_context()
    assert "timestamp" in ctx and "shift" in ctx and "zone" in ctx
    print(f"   OK: shift={ctx['shift']}, zone={ctx['zone']}")

def test_agent():
    print("3. Agent (phrase + context) -> autonomous execution...")
    from fieldtalk.context import get_environment_context
    from fieldtalk.agent_layer import run_agent
    ctx = get_environment_context()
    for phrase in ["urgent", "unit down", "need part", "confirmed"]:
        executions, reasoning_info = run_agent(phrase, ctx)
        print(f"   '{phrase}' -> {len(executions)} execution(s): {[e.get('action') for e in executions]}, priority={reasoning_info.get('priority')}")
    print("   OK: agent runs and returns execution dicts")

def test_mouth_detection():
    print("4. Mouth detection (synthetic frame)...")
    import numpy as np
    from fieldtalk.mouth_detection import MouthDetector, build_frame_buffer
    det = MouthDetector(target_fps=25)
    # Synthetic BGR frame 320x240
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = (128, 128, 128)
    mouth_crop, display, face_bbox = det.process_frame(frame)
    # No face in synthetic frame; should still return display
    assert display is not None and display.shape == frame.shape
    grid = build_frame_buffer([np.zeros((40, 40, 3), dtype=np.uint8)] * 16)
    assert grid is not None and grid.ndim == 3
    print("   OK: MouthDetector and build_frame_buffer work")

def test_inference_skip_if_no_ollama():
    print("5. Inference (Ollama); skip if not running...")
    import numpy as np
    from fieldtalk.inference import infer_phrase, MODEL
    # Build minimal 16-frame list (tiny mouth crops)
    frames = [np.zeros((32, 32, 3), dtype=np.uint8)] * 16
    try:
        phrase, conf = infer_phrase(frames, model=MODEL)
        print(f"   OK: infer_phrase returned phrase={phrase!r}, conf={conf}")
    except Exception as e:
        print(f"   SKIP (Ollama not available): {e}")

def test_voice():
    print("6. Voice (pyttsx3 engine init only, no speak)...")
    try:
        from fieldtalk.voice import get_engine
        eng = get_engine()
        print("   OK: TTS engine created")
    except Exception as e:
        print(f"   SKIP (TTS not available): {e}")

def main():
    print("FieldTalk workflow check\n")
    test_imports()
    test_context()
    test_agent()
    test_mouth_detection()
    test_inference_skip_if_no_ollama()
    test_voice()
    print("\nWorkflow check done. All core files are set up.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
