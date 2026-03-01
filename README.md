# FieldTalk

On-device lip reading industrial communication agent. **Zero cloud support — runs fully offline and independent.** All inference and execution are local; no API calls, no external services.

## The full picture

```
Webcam → MediaPipe → 16 frames
       ↓
Fine-tuned Gemma 3:4b (vision / partner's model)
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

**End-to-end flow (summary):**
1. **Lip reading** — Webcam → MediaPipe mouth region → 16 frames → fine-tuned Gemma 3:4b → GRID-style phrase.
2. **GRID → industrial** — `phrase_mapper.py` maps GRID words to field commands (e.g. *bin* → *unit down*, *blue* → *urgent*).
3. **Agent** — Fine-tuned MLP + context decides actions (alert, ticket, inventory, log). Sub-100ms.
4. **Execution + voice** — Tools run; pyttsx3 speaks the industrial command.

## Offline / no cloud

- **Mouth detection:** MediaPipe Face Mesh (primary) for reliable real-time mouth region; Haar cascade fallback if MediaPipe is unavailable.
- **Inference:** Ollama on `127.0.0.1` only. Model runs on your machine.
- **Agent reasoning:** Local Ollama (Gemma 3:4b) or **fine-tuned agent** (see below). Decision layer on-device.
- **Execution:** Same tools; invocation driven by LLM or fine-tuned model.
- **Voice:** pyttsx3 local TTS. **UI:** Streamlit local. No cloud.

Run with no internet after models are pulled.

## Does it run end-to-end?

**Yes.** From project root:

```powershell
python -m streamlit run app.py
```

Then open http://localhost:8501, click **Start**, and allow webcam access. You get:

1. Live webcam with mouth region highlighted (MediaPipe).
2. After 16 mouth frames, vision model returns a phrase; agent (LLM or fine-tuned) decides actions; tools run; TTS speaks the phrase.
3. Transcription, reasoning, and autonomous actions appear in the panels; latency at the bottom.

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
| **Fine-tuned agent** | **Recommended for demo.** Run `python scripts/finetune_agent.py` once; set `FINE_TUNED_AGENT_PATH=models/finetuned_agent.pt` to use the fine-tuned decision layer. Without it, the app uses base Gemma 3 (Ollama). |

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

### Fine-tuned component (hard requirement for demo)

The app includes a **fine-tuned agent** so the decision layer is not just base Gemma:

1. **Synthetic data:** `data/synthetic_agent_data.jsonl` — (phrase, context) → priority, actions, reasoning.
2. **Fine-tune script:** `python scripts/finetune_agent.py` — trains a small PyTorch MLP on that data and saves `models/finetuned_agent.pt`.
3. **Use in app:** Set `FINE_TUNED_AGENT_PATH` to the saved path (e.g. `models/finetuned_agent.pt`). The app loads it and uses it for action decisions; reasoning text shows "Fine-tuned agent decision".

```powershell
# From project root (install PyTorch for fine-tuning and loading the fine-tuned model)
pip install torch
python scripts/finetune_agent.py
$env:FINE_TUNED_AGENT_PATH = "models/finetuned_agent.pt"
python -m streamlit run app.py
```

Same tools and workflow; the **decision** comes from the fine-tuned model trained on our synthetic (phrase + context) → actions data.

#### What did fine-tuning actually improve?

**Side-by-side (same phrase, e.g. "unit down"):**

|                         | Decision time   |
|-------------------------|-----------------|
| **Base Gemma 3**         | 8–12 seconds    |
| **Fine-tuned MLP**       | **&lt;100 ms**  |

That *is* the fine-tuning story: **speed + determinism** for safety-critical decisions.

> *"We fine-tuned a lightweight decision model for sub-100ms autonomous action selection in safety-critical environments. The base LLM handles complex reasoning. The fine-tuned model handles time-critical execution. That's the production architecture — fast where speed matters, intelligent where reasoning matters."*

The app shows **Decision: Fine-tuned MLP — X ms** (or **Base Gemma 3 — X ms**) in the UI so judges see the difference live.

#### Wire merged Gemma into Ollama (GGUF — clean demo story)

**Full steps (VM → GGUF → Ollama → app):** see **[docs/SETUP_FINETUNED_AGENT.md](docs/SETUP_FINETUNED_AGENT.md)** — understanding agent setup and where the partner’s GRID lip-reading model plugs in (`MODEL`).

After fine-tuning with `finetune_gemma_agent.py`, you get a merged model and (optionally) a GGUF. To use the **merged Gemma** as the agent in the app:

1. **Create Ollama model from GGUF** (from project root):
   ```powershell
   # If fieldtalk_agent_gguf/ contains a single .gguf, Ollama may accept the folder.
   # Otherwise list the dir and set FROM in Modelfile.agent to the exact file.
   ollama create fieldtalk-agent -f Modelfile.agent
   ```
2. **Point the app at it**:
   ```powershell
   $env:AGENT_MODEL = "fieldtalk-agent"
   python -m streamlit run app.py
   ```
   `reasoning.py` uses `AGENT_MODEL` for the Ollama chat; when it’s your custom model name, the **merged Gemma** is the decision layer.

#### 200-example retrain (e.g. on a VM)

Use the industrial safety 200-example set for training:

```powershell
# Optional: explicit path
$env:FIELDTALK_AGENT_DATA = "industrial_safety_200.jsonl"
python finetune_gemma_agent.py
```

If `industrial_safety_200.jsonl` is in the project root, the script picks it automatically when `FIELDTALK_AGENT_DATA` is not set. Then export to GGUF and load in Ollama as above. The 200-example data includes `notify_workers`; the app maps it to `trigger_alert`.

---

### Rehearse ×3 — win the presentation

1. **Run the app** with vision model + agent (MLP or merged Gemma) and confirm webcam → phrase → command → actions → voice.
2. **Use the sidebar**: pick **Demo scenario** (e.g. Emergency), click **Run scenario** to drive the pipeline without lip reading; use **Manual phrase override** if something fails.
3. **Rehearse the full flow at least 3 times** (webcam once, Run scenario once, manual override once) so the demo is bulletproof.

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

1. **Mouth detection** — MediaPipe Face Mesh (primary) for mouth landmarks; Haar cascade fallback. Cropped mouth frames at ~25 fps.
2. **Inference** — 16 mouth frames sent as a 4×4 grid to Ollama (Gemma 3 vision, e.g. `gemma3:4b`). Prompt asks for one phrase from the fixed vocabulary and a confidence score.
3. **Agent triggers** — `urgent`/`evacuate` → `trigger_alert()`; `unit down` → `create_ticket()`; `need part` → `query_inventory()`; `confirmed` → `log_confirmation()`. Each prints action + timestamp.
4. **Voice** — pyttsx3 speaks the transcribed phrase at 150 WPM through default audio (e.g. headphones).
5. **Streamlit UI** — Left: live webcam with mouth highlighted; center: transcription + confidence; right: autonomous actions with timestamps; bottom: latency (frame → voice) in ms.

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
