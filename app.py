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
from fieldtalk.inference import infer_phrase, MODEL, PHRASES as INDUSTRIAL_PHRASES_LIST
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
        st.session_state.transcriptions = []  # list of (grid_phrase, industrial_phrase, confidence, ts_str)
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
    if "demo_scenario" not in st.session_state:
        st.session_state.demo_scenario = "Normal"
    if "manual_override" not in st.session_state:
        st.session_state.manual_override = False
    if "manual_phrase" not in st.session_state:
        st.session_state.manual_phrase = INDUSTRIAL_PHRASES_LIST[0] if INDUSTRIAL_PHRASES_LIST else "check"


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
    # When manual override is on, bypass lip reading; phrase is sent via sidebar "Send phrase"
    if not st.session_state.get("manual_override", False) and len(mouth_buffer) >= 16:
        t0 = time.perf_counter()
        frames = list(mouth_buffer)
        result = infer_phrase(frames, model=MODEL_NAME)
        grid_phrase = result.get("grid_phrase", "unknown")
        industrial_phrase = result.get("industrial_phrase", "check")
        confidence = result.get("confidence", 0.0)
        context = get_environment_context(st.session_state.get("demo_scenario"))
        executions, reasoning_info = run_agent(industrial_phrase, context)
        st.session_state.reasoning_history.append(reasoning_info)
        if st.session_state.tts_engine:
            try:
                speak(industrial_phrase, engine=st.session_state.tts_engine)
            except Exception:
                pass
        t1 = time.perf_counter()
        st.session_state.latency_ms = (t1 - t0) * 1000
        st.session_state.transcriptions.append((grid_phrase, industrial_phrase, confidence, _ts_str()))
        for ex in executions:
            st.session_state.actions.append(ex)
        mouth_buffer.clear()
    time.sleep(0.04)  # ~25 fps
    st.rerun()


# Demo scenario colors (green / orange / red)
SCENARIO_COLORS = {"Normal": "#22c55e", "High Risk": "#f97316", "Emergency": "#ef4444"}


def main():
    st.set_page_config(page_title="FieldTalk", layout="wide", initial_sidebar_state="expanded")
    _init_state()

    # Sidebar: demo scenario + manual phrase override
    with st.sidebar:
        st.subheader("Demo scenario")
        from fieldtalk.context import DEMO_SCENARIOS
        scenario_options = ["Normal", "High Risk", "Emergency"]
        selected = st.selectbox(
            "Context preset",
            scenario_options,
            index=scenario_options.index(st.session_state.demo_scenario) if st.session_state.demo_scenario in scenario_options else 0,
            key="demo_scenario_select",
        )
        st.session_state.demo_scenario = selected
        color = SCENARIO_COLORS.get(selected, "#6b7280")
        st.markdown(
            f'<span style="background: {color}; color: white; padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: 600;">{selected}</span>',
            unsafe_allow_html=True,
        )
        ctx = DEMO_SCENARIOS.get(selected, {})
        st.caption(f"shift={ctx.get('shift', '—')}, temp={ctx.get('temperature', '—')}°C")
        st.caption(f"tickets={ctx.get('active_tickets', '—')}, workers={ctx.get('nearby_workers', '—')}")
        # One-click demo: run pipeline with scenario's canonical phrase (no webcam needed)
        scenario_phrase = {"Normal": "confirmed", "High Risk": "need part", "Emergency": "evacuate"}.get(selected, "check")
        if st.button("Run scenario", key="sidebar_run_scenario", help=f"Send phrase «{scenario_phrase}» and run agent + TTS"):
            context = get_environment_context(selected)
            executions, reasoning_info = run_agent(scenario_phrase, context)
            st.session_state.reasoning_history.append(reasoning_info)
            if st.session_state.tts_engine:
                try:
                    speak(scenario_phrase, engine=st.session_state.tts_engine)
                except Exception:
                    pass
            st.session_state.transcriptions.append((f"[scenario: {selected}]", scenario_phrase, 1.0, _ts_str()))
            for ex in executions:
                st.session_state.actions.append(ex)
            st.rerun()

        st.markdown("---")
        st.subheader("Controls")
        manual_override = st.toggle("Manual phrase override", value=st.session_state.manual_override, key="sidebar_manual_override")
        st.session_state.manual_override = manual_override
        if manual_override:
            phrase_options = INDUSTRIAL_PHRASES_LIST if INDUSTRIAL_PHRASES_LIST else ["check"]
            idx = phrase_options.index(st.session_state.manual_phrase) if st.session_state.manual_phrase in phrase_options else 0
            st.session_state.manual_phrase = st.selectbox(
                "Industrial phrase",
                options=phrase_options,
                index=idx,
                key="sidebar_manual_phrase",
            )
            if st.button("Send phrase", type="primary", key="sidebar_send_phrase"):
                context = get_environment_context(st.session_state.get("demo_scenario"))
                executions, reasoning_info = run_agent(st.session_state.manual_phrase, context)
                st.session_state.reasoning_history.append(reasoning_info)
                if st.session_state.tts_engine:
                    try:
                        speak(st.session_state.manual_phrase, engine=st.session_state.tts_engine)
                    except Exception:
                        pass
                st.session_state.transcriptions.append(
                    ("[manual]", st.session_state.manual_phrase, 1.0, _ts_str())
                )
                for ex in executions:
                    st.session_state.actions.append(ex)
                st.rerun()
            st.markdown(
                '<span style="color: #dc2626; font-weight: 700; font-size: 0.9rem;">MANUAL OVERRIDE</span>',
                unsafe_allow_html=True,
            )

    st.title("FieldTalk")
    st.caption("Lip reading → GRID phrase → phrase_mapper → industrial command → Agent MLP → execution + voice · Offline only")
    st.caption("Use the **sidebar (▶)** for Demo scenario and **Manual phrase override**.")

    with st.expander("The full picture", expanded=False):
        st.markdown("""
```
Webcam → MediaPipe → 16 frames
       ↓
Fine-tuned Gemma 3:4b (partner's model)
       ↓
"bin blue at f two now"   ← GRID phrase
       ↓
phrase_mapper.py
       ↓
"unit down"   ← industrial command
       ↓
Agent MLP + context
       ↓
CRITICAL alert + create ticket + notify workers
       ↓
pyttsx3 voice output
```
        """)

    # Show manual override state in main area when sidebar may be collapsed
    if st.session_state.get("manual_override"):
        st.markdown(
            '<span style="color: #dc2626; font-weight: 700; font-size: 0.9rem;">MANUAL OVERRIDE</span> — phrase sent from sidebar.',
            unsafe_allow_html=True,
        )

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
        scenario = st.session_state.get("demo_scenario", "Normal")
        sc_color = SCENARIO_COLORS.get(scenario, "#6b7280")
        st.markdown(f'Scenario: <span style="background: {sc_color}; color: white; padding: 0.15rem 0.4rem; border-radius: 4px; font-size: 0.85rem;">{scenario}</span>', unsafe_allow_html=True)
        ctx = get_environment_context(scenario)
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
        for row in reversed(st.session_state.transcriptions[-20:]):
            if len(row) == 4:
                grid_phrase, industrial_phrase, conf, ts = row
            else:
                grid_phrase = industrial_phrase = row[0] if row else "—"
                conf = row[1] if len(row) > 1 else 0.0
                ts = row[2] if len(row) > 2 else ""
            st.caption(f"Detected: {grid_phrase}")
            st.markdown(
                f'<span style="color: #22c55e; font-size: 1.25rem; font-weight: 700;">Command: {industrial_phrase.upper()}</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"{conf:.0%} — `{ts}`")
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
