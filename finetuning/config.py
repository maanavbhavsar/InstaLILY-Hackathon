"""
Configuration constants for the GRID lip-reading fine-tuning pipeline.
Single source of truth for paths, model params, grid layout, and vocabulary.
"""
from pathlib import Path

# -- Paths --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "grid"
RAW_VIDEO_DIR = DATA_DIR / "raw"
FRAMES_DIR = DATA_DIR / "frames"
MOUTH_ROI_DIR = DATA_DIR / "mouth_rois"
MOSAIC_DIR = DATA_DIR / "mosaics"
ALIGNMENT_DIR = DATA_DIR / "alignments"
DATASET_DIR = PROJECT_ROOT / "data"
ADAPTER_DIR = PROJECT_ROOT / "models" / "gemma3n-lipreader-lora"

# -- GRID Corpus --
ZENODO_RECORD_ID = "3625687"
ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"
DEFAULT_SPEAKERS = ["s1", "s2", "s3", "s4", "s5"]
ALL_SPEAKERS = [f"s{i}" for i in range(1, 35)]  # s1..s34

# -- Mosaic Layout --
MOSAIC_GRID_ROWS = 3
MOSAIC_GRID_COLS = 4
MOSAIC_CELL_WIDTH = 100   # pixels
MOSAIC_CELL_HEIGHT = 80   # pixels
N_SAMPLE_FRAMES = MOSAIC_GRID_ROWS * MOSAIC_GRID_COLS  # 12

# -- Video Properties --
GRID_VIDEO_FPS = 25
GRID_VIDEO_TOTAL_FRAMES = 75
GRID_SAMPLES_PER_FRAME = 1000  # alignment sample rate (25000 Hz) / video fps (25)

# -- MediaPipe Mouth Detection --
# Same indices as fieldtalk/mouth_detection.py _MOUTH_INDICES
MEDIAPIPE_MOUTH_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]
MOUTH_PADDING = 0.15  # fraction of mouth bbox to pad

# -- GRID Vocabulary --
GRID_COMMANDS = ["bin", "lay", "place", "set"]
GRID_COLORS = ["blue", "green", "red", "white"]
GRID_PREPOSITIONS = ["at", "by", "in", "with"]
GRID_LETTERS = [c for c in "abcdefghijklmnopqrstuvxyz"]  # excludes W
GRID_DIGITS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
GRID_ADVERBS = ["again", "now", "please", "soon"]

# -- Model --
BASE_MODEL_ID = "google/gemma-3n-E4B-it"

# -- LoRA --
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
LORA_DROPOUT = 0.05

# -- Training --
NUM_EPOCHS = 5
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 2e-4
LOGGING_STEPS = 10
EVAL_STEPS = 100
SAVE_STEPS = 200

# -- Dataset Split --
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42

# -- Prompt (must match at inference time) --
_VOCAB_STR = (
    "Commands: bin, lay, place, set | "
    "Colors: blue, green, red, white | "
    "Prepositions: at, by, in, with | "
    "Letters: a-z (except w) | "
    "Digits: zero, one, two, three, four, five, six, seven, eight, nine | "
    "Adverbs: again, now, please, soon"
)

LIP_READING_SYSTEM_PROMPT = (
    "You are a lip reading transcription system. "
    "Output ONLY the spoken words, nothing else. "
    "No explanations, no punctuation, no formatting. "
    "The sentence is always 6 words in the format: "
    "<command> <color> <preposition> <letter> <digit> <adverb>. "
    f"Vocabulary: {_VOCAB_STR}"
)
LIP_READING_PROMPT = (
    "Read the lips in this mosaic of frames (left to right, top to bottom). "
    "Output only the 6 spoken words."
)
