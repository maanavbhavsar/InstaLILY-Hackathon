"""
Streamlit demo dashboard: video → mosaic → lip reading → phrase mapping → agent reasoning → actions.

Visualises each stage of the FieldTalk pipeline on pre-recorded video files.

Usage:
    uv run streamlit run demo_dashboard.py
"""
from __future__ import annotations

import os
import time

import cv2
import numpy as np
import streamlit as st

# Ensure Ollama client uses localhost only (must be before import)
os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")

from pathlib import Path

from demo_inference import build_mosaic_from_rois, extract_frames, run_ollama
from fieldtalk.agent_layer import run_agent
from fieldtalk.context import get_environment_context
from fieldtalk.phrase_mapper import map_to_industrial
from finetuning.build_mosaics import sample_frame_indices
from finetuning.config import N_SAMPLE_FRAMES
from finetuning.extract_mouth_rois import _get_face_mesh, extract_mouth_roi
from finetuning.parse_alignments import parse_alignment_file

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FieldTalk Demo",
    page_icon="🏭",
    layout="wide",
)

st.title("FieldTalk Demo — Lip Reading → Agent Pipeline")

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Pipeline Settings")
    video_path = st.text_input("Video path", value="demo_videos/bbaf2n.mpg")
    align_path = st.text_input("Alignment file (optional)", value="demo_videos/bbaf2n.align")
    model_name = st.text_input("Model name", value="fieldtalk-lipreader")
    scenario = st.selectbox("Scenario", ["Normal", "High Risk", "Emergency"], index=0)
    run_btn = st.button("Run Pipeline", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------
if not run_btn:
    st.info("Configure settings in the sidebar and click **Run Pipeline** to start.")
    st.stop()

pipeline_t0 = time.perf_counter()

# ── Stage 1: Input ─────────────────────────────────────────────────────────
st.subheader("Stage 1 — Input Video")
col_input, col_mosaic = st.columns(2)

with col_input:
    with st.spinner("Extracting frames..."):
        frames = extract_frames(video_path)
        total_frames = len(frames)

    if total_frames == 0:
        st.error("No frames extracted from video.")
        st.stop()

    mid = total_frames // 2
    sample_bgr = frames[mid]
    st.image(cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB), caption=f"Sample frame (#{mid})", use_container_width=True)
    st.metric("Frames extracted", total_frames)

    # Parse alignment
    align_info = None
    if align_path and Path(align_path).exists():
        align_info = parse_alignment_file(Path(align_path), total_frames=total_frames)
        st.write(f"**Speech region:** frames {align_info.speech_start_frame}–{align_info.speech_end_frame}")
        st.write(f"**Ground truth:** \"{align_info.transcription}\"")

# ── Stage 2: Preprocessing ─────────────────────────────────────────────────
with col_mosaic:
    st.subheader("Stage 2 — Preprocessing")

    speech_start = align_info.speech_start_frame if align_info else 0
    speech_end = align_info.speech_end_frame if align_info else total_frames - 1

    indices = sample_frame_indices(speech_start, speech_end, N_SAMPLE_FRAMES)
    indices = [min(i, total_frames - 1) for i in indices]

    with st.spinner("Extracting mouth ROIs..."):
        face_mesh = _get_face_mesh()
        rois = []
        for idx in indices:
            roi = extract_mouth_roi(frames[idx], face_mesh)
            rois.append(roi)
        face_mesh.close()

    detected = sum(1 for r in rois if r is not None)
    mosaic = build_mosaic_from_rois(rois)
    st.image(cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB), caption="3×4 Mouth ROI Mosaic", use_container_width=True)
    st.metric("Mouths detected", f"{detected}/{len(indices)}")

    if detected == 0:
        st.error("No mouths detected — cannot proceed.")
        st.stop()

st.divider()

# ── Stage 3 & 4: Lip Reading + Phrase Mapping ──────────────────────────────
col_lip, col_phrase = st.columns(2)

with col_lip:
    st.subheader("Stage 3 — Lip Reading")
    with st.spinner(f"Running inference ({model_name})..."):
        transcription = run_ollama(mosaic, model_name)
    st.write(f"**GRID transcription:** {transcription}")
    if align_info:
        st.write(f"**Ground truth:** {align_info.transcription}")

with col_phrase:
    st.subheader("Stage 4 — Phrase Mapping")
    industrial_phrase = map_to_industrial(transcription)
    st.markdown(f"### {industrial_phrase.upper()}")
    st.caption("Mapped industrial command")

st.divider()

# ── Stage 5 & 6: Agent Reasoning + Actions ─────────────────────────────────
col_agent, col_actions = st.columns(2)

with col_agent:
    st.subheader("Stage 5 — Agent Reasoning")
    context = get_environment_context(scenario)

    with st.spinner("Running agent reasoning..."):
        agent_t0 = time.perf_counter()
        executions, reasoning_info = run_agent(industrial_phrase, context)
        agent_ms = (time.perf_counter() - agent_t0) * 1000

    priority = reasoning_info.get("priority", "medium").upper()
    source = reasoning_info.get("source", "llm")
    source_label = "Fine-tuned MLP" if source != "llm" else "Gemma 3:4b (LLM)"
    decision_ms = reasoning_info.get("decision_ms", agent_ms)

    # Color-coded priority box
    priority_colors = {"LOW": "success", "MEDIUM": "info", "HIGH": "warning", "CRITICAL": "error"}
    box_fn = getattr(st, priority_colors.get(priority, "info"))
    box_fn(f"**Priority:** {priority}")

    st.write(f"**Scenario:** {scenario}")
    st.write(f"**Decision engine:** {source_label}")
    st.write(f"**Decision time:** {decision_ms:.0f} ms")
    st.write(f"**Reasoning:** {reasoning_info.get('reasoning', '—')}")

    st.caption(
        f"shift={context.get('shift')}, zone={context.get('zone')}, "
        f"temp={context.get('temperature')}°C, "
        f"workers={context.get('nearby_workers')}, "
        f"tickets={context.get('active_tickets')}"
    )

with col_actions:
    st.subheader("Stage 6 — Autonomous Actions")
    if executions:
        for ex in executions:
            st.success(f"**{ex.get('action')}** — {ex.get('message')}")
    else:
        st.info("No autonomous actions triggered.")

# ── Bottom metrics bar ──────────────────────────────────────────────────────
st.divider()
total_pipeline_s = time.perf_counter() - pipeline_t0
m1, m2, m3 = st.columns(3)
m1.metric("Total pipeline time", f"{total_pipeline_s:.1f}s")
m2.metric("Decision time", f"{decision_ms:.0f} ms")
m3.metric("Decision source", source_label)
