# InstaLILY Hackathon — FieldTalk: Project Summary & Prompt

## What We Have

**FieldTalk** is an **on-device lip-reading industrial communication agent**. It runs **fully offline** — no cloud, no external APIs. All inference and execution are local.

### Tech Stack
- **Python 3.14+** (pyproject.toml)
- **Streamlit** — local web UI
- **OpenCV** — webcam capture, face/mouth detection (Haar cascade)
- **Ollama** — local vision model (PaliGemma 2) for lip-reading inference
- **pyttsx3** — local TTS (OS engine) for voice output
- **No cloud** — `OLLAMA_HOST` forced to `http://127.0.0.1:11434`

### Implemented Components

1. **Mouth detection** (`fieldtalk/mouth_detection.py`)
   - Face via OpenCV `haarcascade_frontalface_default.xml`
   - Mouth = lower third of face bbox; cropped at ~25 fps
   - `build_frame_buffer()` — 16 mouth frames → 4×4 grid image for the vision model

2. **Lip-reading inference** (`fieldtalk/inference.py`)
   - 16 mouth frames → single 4×4 grid PNG → Ollama (PaliGemma 2)
   - Fixed vocabulary: *stop, start, help, urgent, clear, confirmed, negative, hold, proceed, done, unit down, need part, all clear, stand by, roger, repeat, abort, ready, check, evacuate*
   - Returns `(phrase, confidence)`; parses model output for phrase + 0–100 confidence

3. **Environmental context** (`fieldtalk/context.py`)
   - `get_environment_context()` → `timestamp`, `shift` (day/swing/night by hour), `zone` (placeholder "field")
   - Used so the agent can reason with phrase + context

4. **Agent layer** (`fieldtalk/agent_layer.py`)
   - Takes **phrase + context** and runs **autonomous execution** via tools:
     - `urgent` / `evacuate` → `trigger_alert()`
     - `unit down` → `create_ticket()`
     - `need part` → `query_inventory()`
     - `confirmed` → `log_confirmation()`
   - Each tool returns `{action, status, message, timestamp}`; no external APIs

5. **Voice** (`fieldtalk/voice.py`)
   - pyttsx3 at 150 WPM; speaks transcribed phrase to default audio (e.g. headphones)
   - Skips speaking if phrase is `"unknown"`

6. **Streamlit app** (`app.py`)
   - **Start/Stop** — webcam capture and lip-reading loop
   - **Three-column layout:**
     - **1. Lip reading** — live webcam with mouth region highlighted; model name + 25 fps
     - **2. Phrase + context** — recent transcriptions (phrase, confidence %, timestamp); current context (shift, zone)
     - **3. Autonomous execution** — recent actions with status and timestamps
   - **Latency** — frame → voice in ms
   - Session state: `running`, `cap`, `detector`, `mouth_buffer` (16 frames), `transcriptions`, `actions`, `latest_frame`, `latency_ms`, `tts_engine`

7. **Workflow test** (`run_workflow_test.py`)
   - Tests: imports, context, agent execution, mouth detection (synthetic frame), inference (skips if no Ollama), voice engine init
   - No webcam required for basic sanity check

### Dependencies (pyproject.toml)
- ollama, opencv-python, pyttsx3, streamlit
- faster-whisper, gdown, torch, torchvision, transformers (listed; lip-reading path uses Ollama + OpenCV only)

---

## What We Are Doing

**Goal:** Enable industrial workers to give **hands-free, voice-free commands** in noisy or restricted environments by **lip-reading** from the webcam, then having an **on-device agent** interpret the phrase with environmental context and **autonomously execute** actions (alerts, tickets, inventory, logging) — all **offline**.

**End-to-end flow:**
1. **Lip reading (minimal)** — Webcam → face/mouth crop → 16-frame grid → local vision model (Ollama/PaliGemma 2) → phrase + confidence.
2. **Agent understands phrase + context** — Context (shift, zone, time) is passed with the phrase so the agent can decide what to do.
3. **Autonomous execution** — Agent runs tools (alert, ticket, inventory, log); results are shown in the UI with timestamps.
4. **Voice feedback** — Transcribed phrase is spoken via local TTS (pyttsx3) at 150 WPM.

**Design constraints:**
- **Zero cloud** — all inference and execution on-device; run with no internet after models are pulled.
- **Simple, minimal lip reading** — fixed vocabulary, grid-of-frames input to a single vision model call.
- **Deterministic agent** — phrase → tool mapping (no LLM for decision-making in current implementation).

---

## One-Paragraph Prompt (for AI / onboarding)

**FieldTalk** is an InstaLILY Hackathon project: an **on-device lip-reading industrial communication agent** that runs **fully offline**. The app uses a **webcam** to capture the face, crops the **mouth region** with OpenCV (Haar cascade), buffers **16 mouth frames** into a 4×4 grid, and sends it to **Ollama (PaliGemma 2)** for local vision inference to get a **phrase + confidence** from a fixed vocabulary (e.g. *urgent*, *unit down*, *need part*, *confirmed*). An **agent layer** takes this phrase plus **environmental context** (shift, zone, time) and **autonomously runs tools** (trigger_alert, create_ticket, query_inventory, log_confirmation), with results and timestamps shown in a **Streamlit UI**. **Voice output** is done via **pyttsx3** (local TTS). There are **no cloud or external API calls**; Ollama is bound to localhost. The codebase includes mouth detection, inference, context, agent, voice, and a workflow test script; the main entry point is `streamlit run app.py`.
