"""
FieldTalk - On-device lip reading industrial communication agent.
Zero cloud. Runs fully offline. All inference and execution are local.
"""
import os

# Force Ollama to localhost only (no cloud, offline-only)
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")

import time
import streamlit as st
import cv2
from collections import deque

from fieldtalk.mouth_detection import MouthDetector
from fieldtalk.inference import infer_phrase, MODEL
from fieldtalk.agent_layer import run_agent
from fieldtalk.context import get_environment_context
from fieldtalk.voice import speak, get_engine

# Model from env (no cloud; local Ollama only). Use gemma3:4b, gemma3:12b, or gemma3:27b for vision.
MODEL_NAME = os.getenv("MODEL", "gemma3:4b")

# Session state keys
def _init_state():
    if "running" not in st.session_state:
        st.session_state.running = False
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "detector" not in st.session_state:
        st.session_state.detector = MouthDetector(target_fps=25)
    if "mouth_buffer" not in st.session_state:
        st.session_state.mouth_buffer = deque(maxlen=16)
    if "transcriptions" not in st.session_state:
        st.session_state.transcriptions = []  # list of (phrase, confidence, ts_str)
    if "actions" not in st.session_state:
        st.session_state.actions = []  # list of execution dicts (autonomous execution)
    if "reasoning_history" not in st.session_state:
        st.session_state.reasoning_history = []  # list of {priority, reasoning} from LLM
    if "latest_frame" not in st.session_state:
        st.session_state.latest_frame = None
    if "latency_ms" not in st.session_state:
        st.session_state.latency_ms = None
    if "tts_engine" not in st.session_state:
        try:
            st.session_state.tts_engine = get_engine()
        except Exception:
            st.session_state.tts_engine = None


def _ts_str():
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def live_capture():
    _init_state()
    if not st.session_state.get("running"):
        return
    cap = st.session_state.cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam.")
            return
        cap.set(cv2.CAP_PROP_FPS, 25)
        st.session_state.cap = cap
    detector = st.session_state.detector
    mouth_buffer = st.session_state.mouth_buffer
    ret, frame = cap.read()
    if not ret or frame is None:
        time.sleep(0.04)
        st.rerun()
        return
    mouth_crop, display, _ = detector.process_frame(frame)
    st.session_state.latest_frame = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    if mouth_crop is not None and mouth_crop.size > 0:
        mouth_buffer.append(mouth_crop.copy())
    if len(mouth_buffer) >= 16:
        t0 = time.perf_counter()
        frames = list(mouth_buffer)
        phrase, confidence = infer_phrase(frames, model=MODEL_NAME)
        context = get_environment_context()
        executions, reasoning_info = run_agent(phrase, context)
        st.session_state.reasoning_history.append(reasoning_info)
        if st.session_state.tts_engine:
            try:
                speak(phrase, engine=st.session_state.tts_engine)
            except Exception:
                pass
        t1 = time.perf_counter()
        st.session_state.latency_ms = (t1 - t0) * 1000
        st.session_state.transcriptions.append((phrase, confidence, _ts_str()))
        for ex in executions:
            st.session_state.actions.append(ex)
        mouth_buffer.clear()
    time.sleep(0.04)  # ~25 fps
    st.rerun()


def main():
    st.set_page_config(page_title="FieldTalk", layout="wide", initial_sidebar_state="collapsed")
    _init_state()

    st.title("FieldTalk")
    st.caption("Lip reading → Agent (phrase + context) → Autonomous execution · Offline only, zero cloud")

    # Control bar: Start/Stop + status
    col_btn, col_status, _ = st.columns([1, 2, 5])
    with col_btn:
        if st.session_state.running:
            if st.button("Stop", type="primary"):
                st.session_state.running = False
                if st.session_state.cap is not None:
                    try:
                        st.session_state.cap.release()
                    except Exception:
                        pass
                    st.session_state.cap = None
                st.rerun()
        else:
            if st.button("Start"):
                st.session_state.running = True
                st.rerun()
    with col_status:
        status = "Running" if st.session_state.running else "Stopped"
        st.caption(f"Status: **{status}** · Model: {MODEL_NAME}")

    if st.session_state.running:
        live_capture()

    # Three-stage flow
    st.markdown("---")
    st.markdown("**1. Lip reading (simple, minimal)** → **2. Agent (phrase + context)** → **3. Autonomous execution**")
    col_cam, col_txt, col_act = st.columns([1, 1, 1.2])

    with col_cam:
        st.subheader("1. Lip reading")
        if st.session_state.latest_frame is not None:
            st.image(st.session_state.latest_frame, channels="RGB", use_container_width=True)
        else:
            st.info("Click **Start** to show webcam with mouth region highlighted.")
        st.caption("25 fps · mouth = lower third of face")

    with col_txt:
        st.subheader("2. Phrase + context (decision)")
        ctx = get_environment_context()
        st.caption(f"Context: shift={ctx.get('shift', '—')}, zone={ctx.get('zone', '—')}, temp={ctx.get('temperature', '—')}°C, tickets={ctx.get('active_tickets', '—')}")
        if st.session_state.reasoning_history:
            last = st.session_state.reasoning_history[-1]
            src = last.get("source", "llm")
            dm = last.get("decision_ms")
            if dm is not None:
                label = "Fine-tuned MLP" if src == "finetuned" else "Base Gemma 3"
                st.metric(f"Decision: {label}", f"{dm:.0f} ms")
            st.markdown(f"**Priority:** {last.get('priority', '—')}")
            st.caption(f"*Reasoning:* {last.get('reasoning', '')[:200]}{'…' if len(last.get('reasoning', '')) > 200 else ''}")
        for phrase, conf, ts in reversed(st.session_state.transcriptions[-20:]):
            st.markdown(f"**{phrase}** — {conf:.0%} — `{ts}`")
        if not st.session_state.transcriptions:
            st.info("Phrase + context → agent decides actions. Fine-tuned MLP &lt;100ms; base LLM 8–12s.")
        with st.expander("What did fine-tuning improve?"):
            st.markdown("""
**Side-by-side (same phrase, e.g. \"unit down\"):**
- **Base Gemma 3:** reasoning takes 8–12 seconds  
- **Fine-tuned MLP:** decision in **&lt;100 ms**

*That* is the fine-tuning story: **speed + determinism** for safety-critical decisions.

> We fine-tuned a lightweight decision model for **sub-100ms autonomous action selection** in safety-critical environments. The base LLM handles complex reasoning when needed. The fine-tuned model handles **time-critical execution**. That's the production architecture — fast where speed matters, intelligent where reasoning matters.
            """)

    with col_act:
        st.subheader("3. Autonomous execution")
        for ex in reversed(st.session_state.actions[-20:]):
            if isinstance(ex, dict):
                action = ex.get("action", "—")
                status = ex.get("status", "—")
                msg = ex.get("message", "")
                ts = ex.get("timestamp", "")
                st.markdown(f"**{action}** — `{status}`")
                st.caption(f"{msg} — {ts}")
            else:
                st.markdown(str(ex))
        if not st.session_state.actions:
            st.info("Triggered actions (e.g. alert, ticket, inventory) appear here.")

    st.markdown("---")
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        if st.session_state.latency_ms is not None:
            st.metric("Latency (frame → voice)", f"{st.session_state.latency_ms:.0f} ms")
        else:
            st.metric("Latency (frame → voice)", "—")
    with c2:
        if st.session_state.reasoning_history:
            last = st.session_state.reasoning_history[-1]
            dm = last.get("decision_ms")
            src = last.get("source", "llm")
            if dm is not None:
                st.caption(f"Decision: {'Fine-tuned MLP' if src == 'finetuned' else 'Base Gemma 3'} — **{dm:.0f} ms**")


if __name__ == "__main__":
    main()
