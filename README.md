# FieldTalk

On-device lip reading industrial communication agent. **Zero cloud support — runs fully offline and independent.** All inference and execution are local; no API calls, no external services.

## The full picture

```
Webcam/Video → MediaPipe → 12–16 frames
       ↓
Fine-tuned Gemma 3:4b (vision, GGUF via Ollama)
       ↓
"bin red at a zero please"   ← GRID phrase
       ↓
phrase_mapper.py
       ↓
"unit down"   ← industrial command
       ↓
Gemma 3:4b agent reasoning + context
       ↓
HIGH priority → trigger_alert + create_ticket + query_inventory
       ↓
pyttsx3 voice output
```

**End-to-end flow (summary):**
1. **Lip reading** — Webcam/video → MediaPipe mouth region → 12 sampled frames → 3x4 mosaic → fine-tuned Gemma 3:4b (GGUF) → GRID-style phrase.
2. **GRID → industrial** — `phrase_mapper.py` maps GRID words to field commands (e.g. *bin* → *unit down*, *blue* → *urgent*).
3. **Agent reasoning** — Gemma 3:4b via Ollama reasons over phrase + environment context (shift, zone, temperature, workers, tickets) and decides priority + actions. Hardcoded fallback if LLM is unavailable.
4. **Execution + voice** — Tools run autonomously (alert, ticket, inventory, log); pyttsx3 speaks the industrial command.

## Offline / no cloud

- **Mouth detection:** MediaPipe Face Mesh (primary) for reliable real-time mouth region; Haar cascade fallback if MediaPipe is unavailable.
- **Lip reading inference:** Ollama on `127.0.0.1` only. Fine-tuned Gemma 3:4b GGUF model runs on your machine.
- **Agent reasoning:** Gemma 3:4b via Ollama reasons over phrase + context. Hardcoded phrase→action fallback if LLM is unavailable.
- **Execution:** Autonomous tool execution (alerts, tickets, inventory, logs).
- **Voice:** pyttsx3 local TTS. **UI:** Streamlit local. No cloud.
- **Model management:** Models are loaded sequentially — lip reader unloads before agent model loads to fit in memory on constrained hardware.

Run with no internet after models are pulled.

## Does it run end-to-end?

**Yes.** Two ways to run:

### CLI demo (video file → full agentic pipeline)

```bash
# Lip reading only
python demo_inference.py --video demo_videos/bbaf2n.mpg --align demo_videos/bbaf2n.align

# Full end-to-end with agent reasoning
python demo_inference.py --video demo_videos/bbaf2n.mpg --align demo_videos/bbaf2n.align --agent

# With scenario context and TTS voice
python demo_inference.py --video demo_videos/bbaf2n.mpg --agent --scenario Emergency --speak
```

The `--agent` flag runs the full pipeline: lip reading → phrase mapping → Gemma 3:4b agent reasoning → autonomous execution → optional TTS. The lip-reader model is unloaded before the agent model loads to conserve memory.

### Streamlit UI (live webcam)

```bash
python -m streamlit run app.py
```

Then open http://localhost:8501, click **Start**, and allow webcam access. You get:

1. Live webcam with mouth region highlighted (MediaPipe).
2. After 16 mouth frames, vision model returns a phrase; Gemma 3:4b agent reasons over phrase + context and decides actions; tools run; TTS speaks the phrase.
3. Transcription, reasoning, and autonomous actions appear in the panels; latency at the bottom.
4. Use the **sidebar** for demo scenarios (Normal/High Risk/Emergency) or manual phrase override.

If Ollama or a vision model isn’t available, the app still runs: lip reading falls back, and the agent uses the built-in phrase→action fallback so the UI and pipeline are exercised.

## What you need to run the app

| Requirement | What to do |
|-------------|------------|
| **Python 3.9+** | Use your system Python or the project `.venv` after `python -m uv sync`. |
| **Dependencies** | `python -m uv sync` (or `python -m pip install -e .`). |
| **Ollama** | Install from [ollama.com](https://ollama.com) and have it running locally. |
| **Vision model (Gemma only)** | `ollama pull gemma3:4b` — official Ollama Gemma vision model (4b; or use `gemma3:12b` / `gemma3:27b` for more capacity). |
| **Webcam** | For live lip reading. |
| **Dataset** | **Not required** to run the app. The app uses the webcam only. |
| **Fine-tuned lip reader** | **Recommended.** Load the GGUF model: `ollama create fieldtalk-lipreader -f Modelfile.lipreader` (see GGUF setup below). Without it, base `gemma3:4b` is used for lip reading. |

### Models to download

- **gemma3:4b** (vision, for lip reading): `ollama pull gemma3:4b`  
  Or use **gemma3:12b** / **gemma3:27b** for larger models (need more RAM/VRAM).

`paligemma2` is **not** available on Ollama; use **gemma3** (4b/12b/27b) which supports image input. Set `MODEL` to override (e.g. `$env:MODEL = "gemma3:12b"`).

### Optional: GRID dataset (for evaluation or future fine-tuning)

The [GRID corpus](http://spandh.dcs.shef.ac.uk/gridcorpus/) is a lip-reading dataset. The **current app does not use it**; it only uses the webcam. Download it only if you want to:

- Evaluate lip-reading accuracy on standard data, or  
- Add your own fine-tuning/training later.

**Download one speaker (e.g. s1) video:**

```powershell
# From project root (PowerShell)
Invoke-WebRequest -Uri "http://spandh.dcs.shef.ac.uk/gridcorpus/s1/video/s1.mpg.tar" -OutFile "s1.mpg.tar" -UseBasicParsing
tar -xf s1.mpg.tar
```

### Agent reasoning

The agent decision layer uses **Gemma 3:4b via Ollama** to reason over the industrial phrase and environment context:

- **Input:** Industrial phrase (e.g. "unit down") + context (shift, zone, temperature, maintenance history, active tickets, nearby workers)
- **Output:** Priority (high/medium/low) + actions (trigger_alert, create_ticket, query_inventory, log_confirmation) + reasoning text
- **Fallback:** If Ollama is unavailable or times out (120s), a hardcoded phrase→action mapping is used
- **Timeout:** Configurable via `AGENT_TIMEOUT` env var (default: 120s)

The UI and CLI both show which engine made the decision: **Gemma 3:4b (LLM)** or **Hardcoded fallback**.

### GGUF lip-reader setup

The fine-tuned lip-reader model requires both a model GGUF and a vision projector GGUF:

1. **Files needed** (in `fieldtalk_lipreader_gguf_v2/`):
   - `gemma3-lipreader-Q4_K_M.gguf` (2.3 GB) — the model weights
   - `mmproj-gemma3-lipreader-f16.gguf` (812 MB) — the vision projector

2. **Load into Ollama:**
   ```bash
   ollama create fieldtalk-lipreader -f Modelfile.lipreader
   ```

3. **Modelfile.lipreader** uses `FROM` for the model and `ADAPTER` for the vision projector.

> **Note:** The `ADAPTER` directive (not `PROJECTOR`) is the correct Ollama Modelfile directive for mmproj files. Without the vision projector, Ollama returns "missing data required for image input".

### Optional: fine-tuned agent GGUF

You can also fine-tune Gemma 3 specifically for agent reasoning using the 200-example industrial safety dataset:

```bash
# On a GPU VM
export FIELDTALK_AGENT_DATA="industrial_safety_200.jsonl"
python finetune_gemma_agent.py

# Load into Ollama
ollama create fieldtalk-agent -f Modelfile.agent

# Point the app at it
export AGENT_MODEL="fieldtalk-agent"
```

See **[docs/SETUP_FINETUNED_AGENT.md](docs/SETUP_FINETUNED_AGENT.md)** for full steps.

---

### Rehearse the demo

1. **CLI demo:** `python demo_inference.py --video demo_videos/bbaf2n.mpg --align demo_videos/bbaf2n.align --agent --scenario Emergency` — runs full pipeline in terminal.
2. **Streamlit app:** `python -m streamlit run app.py` — use sidebar for **Demo scenario** (Emergency), **Run scenario**, or **Manual phrase override**.
3. **Rehearse at least 3 times** (CLI once, Run scenario once, manual override once) so the demo is bulletproof.

## Requirements

- Python 3.9+
- Webcam
- [Ollama](https://ollama.com) 0.6+ installed and running **locally** with **Gemma 3** vision (e.g. `gemma3:4b`). Pull once with internet; then run offline.

## Setup

**1. Install dependencies** (from project root):

```powershell
python -m uv sync
```

(If `uv` is on your PATH you can use `uv sync` instead.)

Or with pip only:

```powershell
python -m pip install -e .
```

**2. Pull Gemma vision model in Ollama** (required):

```powershell
ollama pull gemma3:4b
```

(Requires Ollama 0.6+. For more capacity use `gemma3:12b` or `gemma3:27b`.)

**3. (Optional)** Use a different Gemma 3 size:

```powershell
$env:MODEL = "gemma3:12b"
```

## Run

```powershell
python -m uv run streamlit run app.py
```

Or with venv activated:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

- **Start**: begin webcam capture and lip reading.
- **Stop**: stop capture and release webcam.

## Components

1. **Mouth detection** (`fieldtalk/mouth_detection.py`) — MediaPipe Face Mesh (primary) for mouth landmarks; Haar cascade fallback. Cropped mouth frames at ~25 fps.
2. **Lip reading inference** (`fieldtalk/inference.py`, `demo_inference.py`) — Mouth frames arranged as a grid image, sent to Ollama (fine-tuned Gemma 3:4b GGUF or base). Returns GRID-style phrase + confidence.
3. **Phrase mapping** (`fieldtalk/phrase_mapper.py`) — Maps GRID corpus words to 20 industrial commands (e.g. *bin* → *unit down*, *blue* → *urgent*).
4. **Environment context** (`fieldtalk/context.py`) — Provides shift, zone, temperature, maintenance history, tickets, nearby workers. Three preset demo scenarios (Normal, High Risk, Emergency).
5. **Agent reasoning** (`fieldtalk/reasoning.py`) — Gemma 3:4b via Ollama reasons over phrase + context → priority + actions + reasoning text. Hardcoded fallback if LLM unavailable.
6. **Autonomous execution** (`fieldtalk/agent_layer.py`) — Executes actions: `trigger_alert`, `create_ticket`, `query_inventory`, `log_confirmation`. Each logs action + timestamp.
7. **Voice** (`fieldtalk/voice.py`) — pyttsx3 speaks the industrial command at 150 WPM through default audio.
8. **Streamlit UI** (`app.py`) — Left: live webcam with mouth highlighted; center: transcription + agent reasoning; right: autonomous actions with timestamps; bottom: latency.
9. **CLI demo** (`demo_inference.py`) — Video file → full pipeline with `--agent` flag. Supports `--scenario`, `--speak`, `--save-mosaic`. Unloads lip-reader before loading agent model to conserve memory.

## Vocabulary

Phrases: stop, start, help, urgent, clear, confirmed, negative, hold, proceed, done, unit down, need part, all clear, stand by, roger, repeat, abort, ready, check, evacuate.

---

## Lip Reading Fine-Tuning (Gemma 3n E4B + LoRA on GRID Corpus)

Beyond the base Ollama-powered lip reading, this project includes a full fine-tuning pipeline to train Gemma 3n E4B on the GRID Audiovisual Speech Corpus — producing a model that can actually read lips from mouth-region frame mosaics.

See [`finetuning/IMPLEMENTATION_PLAN.md`](finetuning/IMPLEMENTATION_PLAN.md) for the full runbook with detailed instructions, test commands, and troubleshooting.

### Architecture

```
Camera/Video → MediaPipe Face Mesh → Mouth ROI Crop → 12 Sampled Frames → 3×4 Mosaic (400×240) → Fine-Tuned Gemma 3n E4B + LoRA → Transcription
```

### Quick Start

**1. Data preprocessing (local, CPU-only):**

```bash
# Install dependencies
pip install opencv-python-headless mediapipe==0.10.14 numpy Pillow

# Run full pipeline for 1 speaker (~7 min)
python3 -m finetuning.run_pipeline --speakers s1

# Or for all 5 speakers (~30 min, ~2.1 GB download)
python3 -m finetuning.run_pipeline --speakers s1 s2 s3 s4 s5
```

This downloads videos from Zenodo, extracts frames, crops mouth regions with MediaPipe, builds 3x4 mosaics, and creates train/val/test JSONL splits.

**2. Training (RTX 6000 or any GPU with 24+ GB VRAM):**

```bash
# Prerequisites: accept Gemma license on HuggingFace, then login
huggingface-cli login

# Install GPU dependencies
pip install -r training/requirements.txt

# Smoke test (verify everything works)
python3 training/train.py --max-steps 2 --batch-size 1

# Full training (~1-2 hours on RTX 6000)
python3 training/train.py
```

**3. Evaluation:**

```bash
# Evaluate on test set
python3 training/evaluate.py --adapter-dir models/gemma3n-lipreader-lora/final-adapter

# Test 4-bit quantized inference (simulates laptop deployment)
python3 training/evaluate.py --adapter-dir models/gemma3n-lipreader-lora/final-adapter --quantize-4bit
```

### Project Structure

```
finetuning/                          # Data preprocessing pipeline
├── config.py                        # Central configuration (paths, model params, vocabulary)
├── download_grid.py                 # Download GRID corpus from Zenodo
├── extract_frames.py                # Video → individual PNG frames
├── extract_mouth_rois.py            # MediaPipe Face Mesh → mouth ROI crops
├── parse_alignments.py              # .align file parser (speech boundaries + transcriptions)
├── build_mosaics.py                 # 12-frame sampling → 3×4 mosaic (400×240)
├── format_dataset.py                # Build JSONL dataset with train/val/test splits
├── validate_dataset.py              # Pre-training sanity checks
├── run_pipeline.py                  # Orchestrate all preprocessing steps
├── IMPLEMENTATION_PLAN.md           # Full runbook with test commands
└── LipAgent_Project_Overview (1).md # Original spec document

training/                            # GPU training pipeline
├── requirements.txt                 # GPU dependencies (torch, transformers, trl, peft, etc.)
├── setup_env.sh                     # RTX 6000 environment setup
├── dataset.py                       # Multimodal dataset loader for SFTTrainer
├── train.py                         # LoRA fine-tuning with SFTTrainer
└── evaluate.py                      # WER + per-position accuracy evaluation

data/                                # (gitignored) Generated data
├── grid/raw/{speaker}/*.mpg         # Downloaded GRID videos
├── grid/frames/{speaker}/{vid}/     # Extracted PNG frames
├── grid/mouth_rois/{speaker}/{vid}/ # Cropped mouth regions
├── grid/mosaics/{speaker}/{vid}.jpg # 3×4 mosaic images (400×240)
├── grid/alignments/{speaker}/*.align# Word-level timing files
├── grid_dataset_train.jsonl         # Training data
├── grid_dataset_val.jsonl           # Validation data
└── grid_dataset_test.jsonl          # Test data
```

### Model & Training Details

| Property | Value |
|----------|-------|
| Base model | Gemma 3n E4B (`google/gemma-3n-E4B-it`) |
| Fine-tuning method | LoRA (r=16, alpha=32, targets=q/k/v/o_proj) |
| Training precision | bf16 (full, on 96GB VRAM) |
| Inference precision | 4-bit quantized (~5-6 GB on laptop) |
| Dataset | GRID corpus, 5 speakers, ~4000 train / ~500 val / ~500 test |
| Epochs | 5, batch 8, grad_accum 2, lr 2e-4, cosine schedule |
| Expected accuracy | 40-60% word-level on seen speakers |

### Pipeline Test Results (s1 speaker, 1000 videos)

| Step | Result | Time |
|------|--------|------|
| Download | 1000 videos + 34000 alignments | ~25s |
| Frame extraction | 74,995 frames (75/video) | 47.6s |
| Mouth ROI extraction | 99.8% face detection rate | 347.6s |
| Mosaic building | 1000/1000 built | 0.8s |
| Dataset formatting | 800 train / 100 val / 100 test | <0.1s |
| Validation | All samples passed | 0.3s |
