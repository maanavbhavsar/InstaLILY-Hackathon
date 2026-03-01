# LipAgent: Edge-Native Lip Reading with Fine-Tuned Gemma 3n

## Problem Statement

Communication in high-noise industrial environments (factory floors, construction sites, airport tarmacs) is unreliable. Workers struggle to convey critical information verbally, leading to safety incidents, delayed responses, and operational inefficiency. Existing solutions like push-to-talk radios and bone conduction headsets degrade in extreme noise or require hands-free operation that isn't always possible.

**LipAgent** solves this by fine-tuning a multimodal language model to read lips from video input, transcribe spoken phrases, and feed that transcription into a downstream agentic system for actions like alert generation, ticket logging, or equipment control — all running on-device at the edge.

---

## Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
│  Camera Feed │────▶│  dlib Face   │────▶│  Frame Sampling &    │────▶│  Fine-Tuned Gemma   │
│  (Edge Device)│    │  Detection & │     │  Mosaic Construction │     │  3n E4B + LoRA      │
└─────────────┘     │  Mouth Crop  │     └──────────────────────┘     │  (4-bit, on laptop) │
                     └──────────────┘                                  └────────┬────────────┘
                                                                                │
                                                                         Transcription
                                                                                │
                                                                                ▼
                                                                       ┌─────────────────────┐
                                                                       │  Downstream Agent   │
                                                                       │  (Alert / Ticket /  │
                                                                       │   Command Dispatch) │
                                                                       └─────────────────────┘
```

The system has two stages:

1. **Lip Reading Model** — A fine-tuned Gemma 3n E2B that takes a mosaic of lip-region frames as image input and outputs a text transcription of the spoken phrase.
2. **Downstream Agent** — Consumes the transcription and takes domain-specific actions. This component is out of scope for this document.

---

## Dataset: GRID Corpus

### Overview

The GRID Audiovisual Speech Corpus is a benchmark dataset for lip reading research. It contains 34,000 video recordings from 34 speakers (18 male, 16 female), each speaking 1,000 sentences. Every sentence follows a fixed 6-word grammar:

```
<command> <color> <preposition> <letter> <digit> <adverb>
```

| Position    | Options (count) | Words                              |
|-------------|----------------|------------------------------------|
| Command     | 4              | bin, lay, place, set               |
| Color       | 4              | blue, green, red, white            |
| Preposition | 4              | at, by, in, with                   |
| Letter      | 25             | A–Z (excluding W)                  |
| Digit       | 10             | 0–9                                |
| Adverb      | 4              | again, now, please, soon           |

**Total unique vocabulary: ~51 words.** This constrained grammar makes GRID an ideal fine-tuning target for a hackathon — the model needs to learn a small, closed set of visual-to-text mappings.

### Example

A typical sentence: **"set blue by A two please"**

Each video is 3 seconds long at 25 fps (75 frames), accompanied by a word-level alignment file:

```
0 12250 sil
12250 19250 set
19250 27250 blue
27250 35000 by
35000 42000 A
42000 49500 two
49500 74500 please
```

### Data Source

We use the **original GRID corpus** from Zenodo, starting from raw video files. This is intentional — we build our own preprocessing pipeline (face detection → mouth ROI extraction → frame sampling → mosaic) so that the **exact same pipeline** can be reused during the live demo with a camera feed. No preprocessing shortcuts that would create a train/demo mismatch.

```bash
# Download from https://zenodo.org/records/3625687
# s1.zip through s34.zip — .mpg video files per speaker
# alignments.zip — word-level timing per video
```

---

## Model: Gemma 3n E2B

### Why Gemma 3n

Gemma 3n is Google's on-device multimodal model family, optimized for edge deployment. The E2B variant has 5B raw parameters but runs with an effective memory footprint of ~2B parameters thanks to Per-Layer Embeddings (PLE) and the MatFormer architecture.

| Property              | Gemma 3n E2B     |
|-----------------------|------------------|
| Raw parameters        | 5B               |
| Effective parameters  | ~2B              |
| Min RAM (inference)   | ~2GB             |
| Vision encoder        | MobileNet-V5     |
| Modalities            | Text + Image     |
| Context length        | 32K tokens       |

**Key advantage:** The model accepts image input natively through its MobileNet-V5 vision encoder, which means we can feed lip frame mosaics directly without building a custom vision pipeline.

We train on the **E4B** variant for best quality (we have 96GB VRAM available), then quantize to 4-bit at inference time to fit on an 8GB laptop. The LoRA adapter works across precision levels.

### Why not a dedicated lip reading model?

Dedicated models like LipNet achieve 95%+ accuracy on GRID, but they only transcribe — they can't reason, follow instructions, or integrate into an agentic pipeline. By fine-tuning Gemma 3n, we get a single model that can both lip-read and serve as the language backbone for downstream reasoning, all running on-device.

---

## Data Preparation Pipeline

### Step 1: Video to Frames

Extract raw frames from the `.mpg` video files using OpenCV:

```python
import cv2
import os

def extract_frames(video_path, output_dir):
    """Extract all frames from a GRID .mpg video."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f"frame_{idx:03d}.png"), frame)
        idx += 1
    cap.release()
    return idx
```

### Step 2: Face Detection and Mouth ROI Extraction

We use `dlib` with the 68-point facial landmark predictor to detect the face and crop the mouth region. **This is the same pipeline used during the live demo**, ensuring no train/inference mismatch.

```python
import dlib
import numpy as np
from PIL import Image

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Mouth landmarks are points 48-67 in the 68-point model
MOUTH_POINTS = list(range(48, 68))

def extract_mouth_roi(frame_path, padding=20):
    """Detect face and crop mouth region from a single frame."""
    img = cv2.imread(frame_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None

    shape = predictor(gray, faces[0])
    mouth_coords = np.array([(shape.part(i).x, shape.part(i).y) for i in MOUTH_POINTS])

    x_min = max(0, mouth_coords[:, 0].min() - padding)
    x_max = min(img.shape[1], mouth_coords[:, 0].max() + padding)
    y_min = max(0, mouth_coords[:, 1].min() - padding)
    y_max = min(img.shape[0], mouth_coords[:, 1].max() + padding)

    mouth_crop = img[y_min:y_max, x_min:x_max]
    return Image.fromarray(cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2RGB))
```

> **Note:** Download `shape_predictor_68_face_landmarks.dat` from [dlib's model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). This file is ~100MB and is needed both during training preprocessing and during the live demo.

### Step 3: Subset Selection

For the hackathon, we do not need the full 34,000-video corpus. We select a focused subset:

- **5 speakers** (3 male, 2 female) → ~5,000 videos
- Split: 80% train / 10% validation / 10% test
- This gives us ~4,000 training samples — sufficient for LoRA fine-tuning on a constrained vocabulary task

### Step 4: Frame Sampling

Each GRID video has 75 frames. We sample 12 key frames from the mouth ROIs, skipping the leading and trailing silence using alignment timestamps:

```python
def get_speech_boundaries(align_path, total_frames=75):
    """Parse alignment file and return frame range where speech occurs."""
    speech_start = None
    speech_end = None
    with open(align_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3 and parts[2] != 'sil':
                start_sample = int(parts[0])
                end_sample = int(parts[1])
                if speech_start is None:
                    speech_start = start_sample
                speech_end = end_sample
    # Convert sample indices to frame indices (assuming 25fps, 1000 samples/frame)
    start_frame = max(0, int(speech_start / 1000))
    end_frame = min(total_frames - 1, int(speech_end / 1000))
    return start_frame, end_frame

def sample_mouth_frames(frame_dir, start, end, n_frames=12):
    """Extract mouth ROIs from evenly sampled frames in the speech region."""
    indices = [start + i * (end - start) // (n_frames - 1) for i in range(n_frames)]
    mouth_frames = []
    for idx in indices:
        path = os.path.join(frame_dir, f"frame_{idx:03d}.png")
        if os.path.exists(path):
            roi = extract_mouth_roi(path)
            if roi is not None:
                mouth_frames.append(roi)
    return mouth_frames
```

### Step 5: Mosaic Construction

The 12 sampled frames are stitched into a single 3×4 grid image. This converts the temporal sequence into a spatial layout that Gemma 3n's vision encoder can process.

```python
def create_mosaic(frames, grid=(3, 4), frame_size=(100, 80)):
    """Arrange frames into a grid mosaic image."""
    mosaic = Image.new('RGB', (grid[1] * frame_size[0], grid[0] * frame_size[1]))
    for i, frame in enumerate(frames):
        row, col = divmod(i, grid[1])
        resized = frame.resize(frame_size)
        mosaic.paste(resized, (col * frame_size[0], row * frame_size[1]))
    return mosaic
```

The resulting mosaic is a single image of size 400×240 pixels, read left-to-right, top-to-bottom as a temporal sequence.

### Step 6: Dataset Formatting

Each sample is formatted as a conversation for supervised fine-tuning:

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "mosaics/s1_bbal6n.jpg"},
        {"type": "text", "text": "These frames show a sequence of lip movements from left to right, top to bottom. Transcribe the spoken phrase."}
      ]
    },
    {
      "role": "assistant",
      "content": "bin blue at L six now"
    }
  ]
}
```

The ground truth transcription is extracted directly from the alignment file by concatenating the non-silence words.

---

## Fine-Tuning Process

### Environment

Fine-tuning runs on an **NVIDIA RTX 6000 with 96GB VRAM**. With this much memory, we can use LoRA in full bf16 precision (no need for QLoRA quantization during training), larger batch sizes, and potentially the E4B model. We use QLoRA only for local inference on the 8GB laptop.

### Dependencies

```bash
pip install transformers trl peft bitsandbytes accelerate datasets pillow dlib opencv-python
```

### LoRA Configuration

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

### Training Configuration

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./gemma3n-lipreader",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
)
```

### Training Script

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import get_peft_model

# Load model in full bf16 — 96GB VRAM can handle this easily
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E4B-it",
    torch_dtype="bfloat16",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("google/gemma-3n-E4B-it")

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: ~1-2% of total parameters are trainable

# Train with SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=processor,
)
trainer.train()
```

### Expected Training Time

With ~4,000 samples, batch size 8, and 5 epochs on an RTX 6000 (96GB VRAM): approximately **1-2 hours**.

> **Note:** We train on the **E4B** model for best quality, then quantize to 4-bit at inference time on the laptop. The LoRA adapter is architecture-compatible — the base model just loads in a lower precision.

---

## Exporting and Running Inference Locally

### Step 1: Save the LoRA Adapter

After training completes on the RTX 6000:

```python
# Save only the adapter weights (small, ~10-50MB)
model.save_pretrained("./gemma3n-lipreader-lora")

# Optionally push to HuggingFace Hub
model.push_to_hub("your-username/gemma3n-lipreader-lora")
```

The adapter is lightweight — only the LoRA delta weights are saved, not the full model.

### Step 2: Transfer to Laptop

Copy the adapter directory from the training machine or pull from HuggingFace Hub. You need:

- The adapter weights (`adapter_model.safetensors`, ~10-50MB)
- The adapter config (`adapter_config.json`)
- The base model will be downloaded on first run

### Step 3: Local Inference (8GB RAM Laptop)

The model was trained in bf16 on the RTX 6000, but we load it in 4-bit for inference on the laptop. The LoRA adapter applies on top regardless of the base model's precision.

```python
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image

# Load base model in 4-bit for inference
bnb_config = BitsAndBytesConfig(load_in_4bit=True)

base_model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3n-E4B-it",
    quantization_config=bnb_config,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("google/gemma-3n-E4B-it")

# Load LoRA adapter on top
model = PeftModel.from_pretrained(base_model, "./gemma3n-lipreader-lora")

# Run inference
image = Image.open("test_mosaic.jpg")
prompt = "These frames show a sequence of lip movements from left to right, top to bottom. Transcribe the spoken phrase."

inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=50)
transcription = processor.decode(output[0], skip_special_tokens=True)

print(transcription)
# Expected output: "set red at G nine now"
```

### Live Demo Pipeline

During the demo, the same preprocessing pipeline from training is used on a live camera feed:

```python
import cv2

def live_inference(model, processor, detector, predictor):
    """Capture frames from webcam, build mosaic, run inference."""
    cap = cv2.VideoCapture(0)
    frames_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract mouth ROI using the same dlib pipeline
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            shape = predictor(gray, faces[0])
            mouth_roi = crop_mouth(frame, shape)  # same as extract_mouth_roi
            frames_buffer.append(mouth_roi)

        # Once we have 12 frames, build mosaic and transcribe
        if len(frames_buffer) >= 12:
            mosaic = create_mosaic(frames_buffer[:12])
            inputs = processor(images=mosaic, text=PROMPT, return_tensors="pt")
            output = model.generate(**inputs.to(model.device), max_new_tokens=50)
            transcription = processor.decode(output[0], skip_special_tokens=True)

            # Pass transcription to downstream agent
            print(f"Transcribed: {transcription}")
            frames_buffer = []

    cap.release()
```

### Memory Footprint (Inference on Laptop)

| Component                     | Size       |
|-------------------------------|------------|
| Gemma 3n E4B (4-bit)         | ~3 GB      |
| LoRA adapter                  | ~20-80 MB  |
| Vision encoder (MobileNet-V5) | ~300 MB    |
| dlib face detector + landmarks| ~100 MB    |
| Inference overhead            | ~1-2 GB    |
| **Total**                     | **~5-6 GB**|

This fits comfortably within an 8GB RAM laptop.

---

## What Happens After Transcription

The transcription output (e.g., `"set red at G nine now"`) is passed as a string to the downstream agentic system. That system is responsible for interpreting the command and executing domain-specific actions such as:

- Generating safety alerts
- Logging maintenance tickets
- Dispatching equipment commands
- Requesting confirmation from the user

The interface between the lip reading model and the downstream agent is a simple text string. This clean separation means both components can be developed, tested, and improved independently.

---

## Expected Performance and Limitations

### What we expect to achieve

- **40-60% word-level accuracy** on seen speakers from the GRID subset — a reasonable outcome given that we're repurposing a general-purpose VLM for a specialized temporal task via spatial encoding.
- A **working end-to-end demo** showing: camera input → mosaic → transcription → downstream action, all running on a laptop.

### Known limitations

- **Temporal information loss:** The mosaic approach encodes time as spatial position. The model has no explicit notion of frame ordering beyond the left-to-right, top-to-bottom convention.
- **Constrained vocabulary only:** The model is fine-tuned on GRID's 51-word vocabulary. It will not generalize to open-vocabulary lip reading.
- **Speaker dependency:** Performance will be stronger on speakers seen during training.
- **Not real-time:** Mosaic construction and inference will add latency. This is a proof-of-concept, not a production system.

### What would make this better (beyond hackathon scope)

- Video-native multimodal models that process frame sequences directly
- Larger and more diverse lip reading datasets (LRS2, LRS3)
- Dedicated temporal encoders feeding into the LLM
- Real-time frame processing pipeline with streaming inference
