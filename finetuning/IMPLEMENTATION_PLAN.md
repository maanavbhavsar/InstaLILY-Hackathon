# LipAgent Fine-Tuning: Implementation Plan & Runbook

## Context

**Problem:** The FieldTalk app currently uses a base Gemma 3 model (via Ollama) for lip reading — it has never been fine-tuned on actual lip-reading data. The model guesses from a 20-word industrial vocabulary using raw mouth-region images, with no training signal from real lip movements.

**Solution:** Fine-tune **Gemma 3n E4B** on the **GRID Audiovisual Speech Corpus** using **LoRA**, producing a model that can actually read lips from mouth-region frame mosaics. The GRID corpus has 34,000 videos of 34 speakers each saying 1,000 sentences from a constrained 51-word vocabulary — ideal for proving the concept.

**Architecture:**
```
Camera/Video → MediaPipe Face Mesh → Mouth ROI Crop → 12 Sampled Frames → 3×4 Mosaic (400×240) → Fine-Tuned Gemma 3n E4B + LoRA → Transcription
```

**Hardware:**
- **Local (MacBook/dev machine):** Data preprocessing pipeline (CPU-bound)
- **RTX 6000 (96GB VRAM) via SSH:** Model training (GPU-bound)

**Key Decision — MediaPipe (not dlib):** The spec doc suggests dlib, but the existing app uses MediaPipe Face Mesh. We use MediaPipe for both training preprocessing and live inference to ensure zero train/inference mismatch. The only difference: `static_image_mode=True` for batch processing, `False` for live webcam.

---

## Dataset: GRID Corpus

| Property | Value |
|----------|-------|
| Source | [Zenodo record 3625687](https://zenodo.org/records/3625687) |
| Total videos | 34,000 (34 speakers × 1,000 each) |
| Our subset | 5 speakers (~5,000 videos) |
| Split | 80% train / 10% val / 10% test |
| Sentence format | `<command> <color> <preposition> <letter> <digit> <adverb>` |
| Vocabulary | ~51 words |
| Video length | 3 seconds, 25 fps, 75 frames |
| Alignments | Word-level timing per video (.align files) |

**GRID vocabulary:**
- Commands: bin, lay, place, set
- Colors: blue, green, red, white
- Prepositions: at, by, in, with
- Letters: A-Z (excluding W)
- Digits: zero through nine
- Adverbs: again, now, please, soon

**Example sentence:** "set blue by a two please"

---

## File Structure

```
finetuning/
├── IMPLEMENTATION_PLAN.md          ← This file
├── LipAgent_Project_Overview (1).md  ← Original spec document
├── __init__.py
├── config.py                       ← Central configuration (paths, params)
├── download_grid.py                ← Download GRID corpus from Zenodo
├── extract_frames.py               ← Video → individual frames
├── extract_mouth_rois.py           ← MediaPipe face detection → mouth crop
├── parse_alignments.py             ← .align file parser
├── build_mosaics.py                ← Frame sampling + 3×4 mosaic construction
├── format_dataset.py               ← Build JSONL for SFTTrainer
├── validate_dataset.py             ← Pre-training sanity checks
└── run_pipeline.py                 ← Orchestrate all preprocessing steps

training/
├── requirements.txt                ← GPU training dependencies
├── setup_env.sh                    ← RTX 6000 environment setup
├── dataset.py                      ← PyTorch Dataset for SFTTrainer
├── train.py                        ← LoRA fine-tuning with SFTTrainer
└── evaluate.py                     ← WER + accuracy evaluation

data/
├── grid/                           ← (gitignored) GRID corpus data
│   ├── raw/{speaker}/*.mpg
│   ├── frames/{speaker}/{video_id}/frame_NNN.png
│   ├── mouth_rois/{speaker}/{video_id}/frame_NNN.png
│   ├── mosaics/{speaker}/{video_id}.jpg
│   └── alignments/{speaker}/*.align
├── grid_dataset_train.jsonl        ← Training data
├── grid_dataset_val.jsonl          ← Validation data
└── grid_dataset_test.jsonl         ← Test data
```

---

## Phase 1: Data Pipeline (Local)

### Step 1: Configuration — `finetuning/config.py`

All constants are defined here. Key values:

| Constant | Value | Notes |
|----------|-------|-------|
| `MOSAIC_GRID_ROWS` | 3 | |
| `MOSAIC_GRID_COLS` | 4 | |
| `MOSAIC_CELL_WIDTH` | 100px | |
| `MOSAIC_CELL_HEIGHT` | 80px | |
| `N_SAMPLE_FRAMES` | 12 | = 3 × 4 |
| `MOUTH_PADDING` | 0.15 | Same as app |
| `LORA_R` | 16 | |
| `LORA_ALPHA` | 32 | |
| `BATCH_SIZE` | 8 | |
| `NUM_EPOCHS` | 5 | |
| `LEARNING_RATE` | 2e-4 | |

**Test:**
```bash
python -c "from finetuning.config import *; print(f'Grid: {MOSAIC_GRID_ROWS}x{MOSAIC_GRID_COLS}'); assert MOSAIC_GRID_ROWS * MOSAIC_GRID_COLS == N_SAMPLE_FRAMES; print('OK')"
```

---

### Step 2: Download GRID Corpus — `finetuning/download_grid.py`

Downloads from Zenodo: video zips (~400MB each) + alignments (~20MB).

```bash
# Dry run — verify URLs are reachable without downloading
python -m finetuning.download_grid --speakers s1 --dry-run

# Download single speaker for testing
python -m finetuning.download_grid --speakers s1

# Download all 5 speakers + alignments (~2.1 GB total)
python -m finetuning.download_grid --speakers s1 s2 s3 s4 s5

# Use existing local files instead of downloading
python -m finetuning.download_grid --speakers s1 --local-path /path/to/grid
```

**Test:**
```bash
# Verify downloads
ls -lh data/grid/raw/s1/*.mpg | head -5
ls -lh data/grid/alignments/s1/*.align | head -5
echo "Video count:"; ls data/grid/raw/s1/*.mpg | wc -l
echo "Align count:"; ls data/grid/alignments/s1/*.align | wc -l
# Expected: ~1000 videos, ~1000 alignments per speaker
```

**Troubleshooting:**
- If Zenodo is slow: files are ~400MB each, expect 5-15 min per speaker depending on connection
- If download fails mid-way: re-run the same command, it skips already-extracted speakers
- If zip is corrupt: delete the partial zip and re-run

---

### Step 3: Extract Frames — `finetuning/extract_frames.py`

Opens each `.mpg` video with OpenCV and saves individual frames as PNG.

```bash
# Extract from single speaker (uses parallel workers)
python -m finetuning.extract_frames --speakers s1 --workers 4

# Extract from single video for testing
python -m finetuning.extract_frames --speakers s1 --limit 1

# All 5 speakers
python -m finetuning.extract_frames --speakers s1 s2 s3 s4 s5 --workers 8
```

**Expected output:** `data/grid/frames/{speaker}/{video_id}/frame_000.png` through `frame_074.png` (~75 frames per video)

**Test:**
```bash
python -c "
import os
d = 'data/grid/frames/s1/'
vids = sorted(os.listdir(d))[:1]
for v in vids:
    frames = os.listdir(os.path.join(d, v))
    print(f'{v}: {len(frames)} frames')
    assert 70 <= len(frames) <= 80, f'Unexpected frame count: {len(frames)}'
print('OK')
"
```

**Disk usage:** ~5KB/frame × 75 frames × 1000 videos × 5 speakers ≈ 1.8 GB

---

### Step 4: Extract Mouth ROIs — `finetuning/extract_mouth_rois.py`

Uses MediaPipe Face Mesh (`static_image_mode=True`) with the same landmark indices as the live app (`fieldtalk/mouth_detection.py` line 11).

```bash
# Process one video's frames
python -m finetuning.extract_mouth_rois --speakers s1 --limit 1

# Process all of s1 (sequential — MediaPipe is not thread-safe)
python -m finetuning.extract_mouth_rois --speakers s1

# All speakers
python -m finetuning.extract_mouth_rois --speakers s1 s2 s3 s4 s5
```

**Expected output:** `data/grid/mouth_rois/{speaker}/{video_id}/frame_NNN.png` — cropped mouth regions.

**Test:**
```bash
python -c "
import cv2, os
d = 'data/grid/mouth_rois/s1/'
vids = sorted(os.listdir(d))[:1]
for v in vids:
    rois = [f for f in os.listdir(os.path.join(d, v)) if f.endswith('.png')]
    print(f'{v}: {len(rois)} mouth ROIs')
    img = cv2.imread(os.path.join(d, v, rois[0]))
    h, w = img.shape[:2]
    print(f'  Size: {w}x{h}')
    assert w > 20 and h > 20, f'ROI too small'
print('OK')
"
```

**Troubleshooting:**
- Videos where >20% of frames fail face detection are skipped (logged as warnings)
- GRID videos are controlled lab recordings; MediaPipe should detect faces in 99%+ of frames
- If many videos fail: check that frames were extracted correctly (Step 3)

---

### Step 5: Parse Alignments — `finetuning/parse_alignments.py`

Parses `.align` files to get speech boundaries and ground truth transcriptions.

**Alignment format:**
```
0 12250 sil
12250 19250 set
19250 27250 blue
...
```

**Conversion:** `frame_index = audio_sample / 1000` (25000 Hz audio ÷ 25 fps = 1000 samples/frame)

```bash
# Parse and print first 3 alignments
python -m finetuning.parse_alignments --speakers s1 --print-first 3
```

**Test (no data needed — uses synthetic input):**
```bash
python -c "
from finetuning.parse_alignments import parse_alignment_lines
lines = ['0 12250 sil', '12250 19250 set', '19250 27250 blue', '27250 35000 by', '35000 42000 a', '42000 49500 two', '49500 74500 please']
info = parse_alignment_lines(lines, video_id='test', speaker_id='s1')
assert info.transcription == 'set blue by a two please'
assert info.speech_start_frame == 12
assert info.speech_end_frame == 74
assert len(info.words) == 6
print(f'Transcription: \"{info.transcription}\"')
print(f'Frames: {info.speech_start_frame}-{info.speech_end_frame}')
print('OK')
"
```

---

### Step 6: Build Mosaics — `finetuning/build_mosaics.py`

For each video:
1. Get speech boundaries from alignment (skip silence)
2. Sample 12 evenly-spaced frame indices from the speech region
3. Load corresponding mouth ROIs, resize to 100×80
4. Arrange in 3×4 grid → 400×240 mosaic
5. Save as JPEG (quality 95)

**Sampling formula:** `indices[i] = start + i * (end - start) // (n_frames - 1)` for i in 0..11

```bash
# Build mosaics for s1 (needs Steps 3-5 completed first)
python -m finetuning.build_mosaics --speakers s1

# Build with limit for testing
python -m finetuning.build_mosaics --speakers s1 --limit 5
```

**Expected output:** `data/grid/mosaics/{speaker}/{video_id}.jpg` — 400×240 JPEG images

**Test:**
```bash
python -c "
import cv2, os
d = 'data/grid/mosaics/s1/'
files = [f for f in os.listdir(d) if f.endswith('.jpg')][:1]
for f in files:
    img = cv2.imread(os.path.join(d, f))
    h, w = img.shape[:2]
    print(f'{f}: {w}x{h}')
    assert (w, h) == (400, 240), f'Wrong dims: {w}x{h}'
print('OK')
"
```

**Disk usage:** ~10KB/mosaic × 5000 videos ≈ 50 MB

---

### Step 7: Format Dataset — `finetuning/format_dataset.py`

Pairs mosaics with ground truth transcriptions and splits into train/val/test JSONL files.

**Output format (per line):**
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "data/grid/mosaics/s1/bbaf2n.jpg"},
        {"type": "text", "text": "These frames show a sequence of lip movements from left to right, top to bottom. Transcribe the spoken phrase."}
      ]
    },
    {
      "role": "assistant",
      "content": "bin blue at l six now"
    }
  ]
}
```

```bash
python -m finetuning.format_dataset --speakers s1 s2 s3 s4 s5
```

**Test:**
```bash
python -c "
import json, os
for split in ['train', 'val', 'test']:
    path = f'data/grid_dataset_{split}.jsonl'
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    print(f'{split}: {len(lines)} samples')
    s = lines[0]
    assert 'messages' in s
    assert len(s['messages']) == 2
    print(f'  Sample: \"{s[\"messages\"][1][\"content\"]}\"')
print('OK')
"
```

**Expected sizes (5 speakers):** ~4,000 train / ~500 val / ~500 test

---

### Step 8: Validate Dataset — `finetuning/validate_dataset.py`

Pre-training sanity checks:
- All mosaic images exist and are 400×240
- All transcriptions follow GRID grammar (6 words)
- Split sizes match expectations
- No duplicates

```bash
python -m finetuning.validate_dataset
```

---

### Step 9: Run Full Pipeline — `finetuning/run_pipeline.py`

Orchestrates Steps 2-8 in sequence.

```bash
# Full pipeline for one speaker (for testing)
python -m finetuning.run_pipeline --speakers s1

# Full pipeline for all 5 speakers
python -m finetuning.run_pipeline --speakers s1 s2 s3 s4 s5

# Resume from a specific step
python -m finetuning.run_pipeline --speakers s1 --start-from mosaics

# Skip download (if data already exists)
python -m finetuning.run_pipeline --speakers s1 --skip-download
```

---

## Phase 2: Training Pipeline (RTX 6000 via SSH)

### Prerequisites

1. SSH into the RTX 6000 machine
2. Clone the repo (or rsync the project)
3. Transfer the preprocessed data (`data/grid/mosaics/` + `data/grid_dataset_*.jsonl`)

**Transfer data to RTX 6000:**
```bash
# From local machine:
rsync -avz data/grid/mosaics/ user@rtx6000:~/InstaLILY-Hackathon/data/grid/mosaics/
rsync -avz data/grid_dataset_*.jsonl user@rtx6000:~/InstaLILY-Hackathon/data/
```

### Step 1: Environment Setup — `training/setup_env.sh`

```bash
# On the RTX 6000:
bash training/setup_env.sh
```

This installs dependencies, verifies CUDA, and logs into HuggingFace (Gemma is gated).

**Test:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('google/gemma-3n-E4B-it'); print(f'Model: {c.model_type}')"
```

### Step 2: Training — `training/train.py`

Loads Gemma 3n E4B in bf16, applies LoRA, trains with SFTTrainer.

**Model:** `google/gemma-3n-E4B-it`
**LoRA config:** r=16, alpha=32, targets=q/k/v/o_proj, dropout=0.05
**Training:** 5 epochs, batch 8, grad_accum 2, lr 2e-4, cosine schedule, gradient checkpointing

```bash
# Smoke test (1 step, verify pipeline works)
python training/train.py \
  --dataset-dir ./data \
  --output-dir ./output-smoke \
  --num-epochs 1 \
  --batch-size 1 \
  --max-steps 2

# Overfit test (memorize 10 samples)
python training/train.py \
  --dataset-dir ./data \
  --output-dir ./output-overfit \
  --num-epochs 50 \
  --batch-size 2 \
  --max-samples 10

# Full training (~1-2 hours on RTX 6000)
python training/train.py \
  --dataset-dir ./data \
  --output-dir ./output
```

**Test:**
```bash
ls ./output/final-adapter/
# Should contain: adapter_model.safetensors, adapter_config.json
```

### Step 3: Evaluation — `training/evaluate.py`

Computes WER, sentence accuracy, per-position accuracy on the test set.

```bash
# Full evaluation
python training/evaluate.py \
  --adapter-dir ./output/final-adapter \
  --dataset-dir ./data

# Quick eval on 10 samples
python training/evaluate.py \
  --adapter-dir ./output/final-adapter \
  --dataset-dir ./data \
  --max-samples 10

# Test 4-bit quantized inference (simulates laptop deployment)
python training/evaluate.py \
  --adapter-dir ./output/final-adapter \
  --dataset-dir ./data \
  --quantize-4bit
```

**Expected results:** 40-60% word-level accuracy on seen speakers.

### Step 4: Export Adapter

The adapter is saved automatically by `train.py`. Transfer back to local:

```bash
# From local machine:
rsync -avz user@rtx6000:~/InstaLILY-Hackathon/output/final-adapter/ models/gemma3n-lipreader-lora/
```

The adapter is lightweight (~10-50MB): just `adapter_model.safetensors` + `adapter_config.json`.

---

## Model Details

### Gemma 3n E4B

| Property | Value |
|----------|-------|
| Raw parameters | 5B |
| Effective parameters | ~2B (PLE + MatFormer) |
| Vision encoder | MobileNet-V5 |
| Modalities | Text + Image |
| Context length | 32K tokens |
| Training precision | bf16 (full, on 96GB VRAM) |
| Inference precision | 4-bit quantized (~3GB on laptop) |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 32 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Dropout | 0.05 |
| Trainable params | ~1-2% of total |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 5 |
| Batch size | 8 |
| Gradient accumulation | 2 (effective batch = 16) |
| Learning rate | 2e-4 |
| Scheduler | Cosine |
| Warmup | 5% of steps |
| Precision | bf16 |
| Gradient checkpointing | Yes |

### Memory Footprint (Inference on Laptop)

| Component | Size |
|-----------|------|
| Gemma 3n E4B (4-bit) | ~3 GB |
| LoRA adapter | ~20-80 MB |
| Vision encoder (MobileNet-V5) | ~300 MB |
| Inference overhead | ~1-2 GB |
| **Total** | **~5-6 GB** |

---

## Expected Performance & Limitations

### Expected
- 40-60% word-level accuracy on seen speakers from GRID subset
- Working end-to-end: mosaic → transcription

### Known limitations
- **Temporal info loss:** Mosaic encodes time as spatial position; no explicit frame ordering
- **Constrained vocabulary:** Only GRID's 51 words; no open-vocabulary lip reading
- **Speaker dependent:** Better on seen speakers
- **Not real-time:** Mosaic + inference adds latency

### What would improve this (beyond hackathon)
- Video-native multimodal models (process frame sequences directly)
- Larger datasets (LRS2, LRS3)
- Temporal encoders feeding into the LLM
- Streaming inference pipeline

---

## Quick Reference: Full Pipeline Commands

```bash
# === LOCAL: Data Preprocessing ===

# 1. Download (one-time, ~2.1 GB for 5 speakers)
python -m finetuning.download_grid --speakers s1 s2 s3 s4 s5

# 2. Extract frames
python -m finetuning.extract_frames --speakers s1 s2 s3 s4 s5 --workers 8

# 3. Extract mouth ROIs
python -m finetuning.extract_mouth_rois --speakers s1 s2 s3 s4 s5

# 4. Build mosaics
python -m finetuning.build_mosaics --speakers s1 s2 s3 s4 s5

# 5. Format dataset
python -m finetuning.format_dataset --speakers s1 s2 s3 s4 s5

# 6. Validate
python -m finetuning.validate_dataset

# Or run all at once:
python -m finetuning.run_pipeline --speakers s1 s2 s3 s4 s5

# === TRANSFER TO RTX 6000 ===
rsync -avz data/grid/mosaics/ user@rtx6000:~/project/data/grid/mosaics/
rsync -avz data/grid_dataset_*.jsonl user@rtx6000:~/project/data/

# === RTX 6000: Training ===
bash training/setup_env.sh
python training/train.py --dataset-dir ./data --output-dir ./output

# === RTX 6000: Evaluation ===
python training/evaluate.py --adapter-dir ./output/final-adapter --dataset-dir ./data

# === TRANSFER BACK ===
rsync -avz user@rtx6000:~/project/output/final-adapter/ models/gemma3n-lipreader-lora/
```
