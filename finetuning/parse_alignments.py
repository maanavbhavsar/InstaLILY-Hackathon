from __future__ import annotations

"""
Parse GRID corpus alignment files to extract:
  - Speech boundaries (start/end frame indices, skipping silence)
  - Ground truth transcriptions (concatenated non-silence words)

Alignment format (one line per word):
    0 12250 sil
    12250 19250 set
    19250 27250 blue
    ...

Sample indices are at 25000 Hz audio rate. Video is 25 fps.
So 1 frame = 1000 audio samples.

Usage:
    python -m finetuning.parse_alignments --speakers s1
    python -m finetuning.parse_alignments --speakers s1 --print-first 3
"""
import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

from .config import (
    ALIGNMENT_DIR,
    DEFAULT_SPEAKERS,
    GRID_VIDEO_TOTAL_FRAMES,
    GRID_SAMPLES_PER_FRAME,
)

logger = logging.getLogger(__name__)

# Words to skip in alignment files (silence / short pause)
SILENCE_TOKENS = {"sil", "sp"}


@dataclass
class AlignmentInfo:
    """Parsed alignment for a single GRID video."""

    video_id: str
    speaker_id: str
    words: list[str] = field(default_factory=list)
    transcription: str = ""
    speech_start_frame: int = 0
    speech_end_frame: int = 0
    word_boundaries: list[tuple[int, int, str]] = field(default_factory=list)


def parse_alignment_lines(
    lines: list[str],
    video_id: str = "",
    speaker_id: str = "",
    total_frames: int = GRID_VIDEO_TOTAL_FRAMES,
    samples_per_frame: int = GRID_SAMPLES_PER_FRAME,
) -> AlignmentInfo:
    """
    Parse alignment content lines and return structured AlignmentInfo.

    Each line: '<start_sample> <end_sample> <word>'
    Silence entries ('sil', 'sp') are excluded from the transcription.
    Frame conversion: frame_index = sample_index / samples_per_frame
    """
    words = []
    word_boundaries = []
    speech_start_frame = None
    speech_end_frame = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue

        start_sample = int(parts[0])
        end_sample = int(parts[1])
        word = parts[2]

        start_frame = max(0, int(start_sample / samples_per_frame))
        end_frame = min(total_frames - 1, int(end_sample / samples_per_frame))

        if word.lower() in SILENCE_TOKENS:
            continue

        words.append(word.lower())
        word_boundaries.append((start_frame, end_frame, word.lower()))

        if speech_start_frame is None:
            speech_start_frame = start_frame
        speech_end_frame = end_frame

    if speech_start_frame is None:
        speech_start_frame = 0
    if speech_end_frame is None:
        speech_end_frame = total_frames - 1

    return AlignmentInfo(
        video_id=video_id,
        speaker_id=speaker_id,
        words=words,
        transcription=" ".join(words),
        speech_start_frame=speech_start_frame,
        speech_end_frame=speech_end_frame,
        word_boundaries=word_boundaries,
    )


def parse_alignment_file(
    align_path: Path,
    video_id: str | None = None,
    speaker_id: str | None = None,
    total_frames: int = GRID_VIDEO_TOTAL_FRAMES,
    samples_per_frame: int = GRID_SAMPLES_PER_FRAME,
) -> AlignmentInfo:
    """Parse a single .align file and return structured AlignmentInfo."""
    if video_id is None:
        video_id = align_path.stem
    if speaker_id is None:
        speaker_id = align_path.parent.name

    with open(align_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    return parse_alignment_lines(
        lines,
        video_id=video_id,
        speaker_id=speaker_id,
        total_frames=total_frames,
        samples_per_frame=samples_per_frame,
    )


def parse_all_alignments(
    speakers: list[str],
    alignment_root: Path = ALIGNMENT_DIR,
) -> dict[str, AlignmentInfo]:
    """
    Parse alignment files for all specified speakers.
    Returns dict keyed by '{speaker_id}/{video_id}'.
    """
    results: dict[str, AlignmentInfo] = {}

    for speaker in speakers:
        speaker_dir = alignment_root / speaker
        if not speaker_dir.is_dir():
            logger.warning(f"Alignment directory not found: {speaker_dir}")
            continue

        align_files = sorted(speaker_dir.glob("*.align"))
        if not align_files:
            logger.warning(f"No .align files in {speaker_dir}")
            continue

        for align_path in align_files:
            video_id = align_path.stem
            key = f"{speaker}/{video_id}"
            try:
                info = parse_alignment_file(align_path, video_id=video_id, speaker_id=speaker)
                results[key] = info
            except Exception as e:
                logger.warning(f"Failed to parse {align_path}: {e}")

        logger.info(f"Parsed {len(align_files)} alignments for {speaker}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Parse GRID alignment files")
    parser.add_argument("--speakers", nargs="+", default=DEFAULT_SPEAKERS)
    parser.add_argument("--alignments-dir", type=Path, default=ALIGNMENT_DIR)
    parser.add_argument("--print-first", type=int, default=0, help="Print first N parsed alignments")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    alignments = parse_all_alignments(args.speakers, alignment_root=args.alignments_dir)
    print(f"Parsed {len(alignments)} total alignments")

    if args.print_first > 0:
        for i, (key, info) in enumerate(alignments.items()):
            if i >= args.print_first:
                break
            print(f"  {key}: \"{info.transcription}\" (frames {info.speech_start_frame}-{info.speech_end_frame})")


if __name__ == "__main__":
    main()
