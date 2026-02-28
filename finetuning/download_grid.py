from __future__ import annotations

"""
Download GRID Audiovisual Speech Corpus from Zenodo.
Downloads video archives + alignment files for specified speakers.

Usage:
    python -m finetuning.download_grid --speakers s1 s2 s3 s4 s5
    python -m finetuning.download_grid --speakers s1 --dry-run
    python -m finetuning.download_grid --speakers s1 --local-path /path/to/existing/grid
"""
import argparse
import logging
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from .config import ZENODO_BASE_URL, DEFAULT_SPEAKERS, RAW_VIDEO_DIR, ALIGNMENT_DIR, DATA_DIR

logger = logging.getLogger(__name__)

# Speaker s21 is not available in the GRID corpus on Zenodo
UNAVAILABLE_SPEAKERS = {"s21"}

MAX_RETRIES = 3


def download_file(url: str, dest: Path, retries: int = MAX_RETRIES) -> bool:
    """Download a file with progress reporting and retry logic."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"Downloading {url} (attempt {attempt}/{retries})")
            req = urllib.request.Request(url, headers={"User-Agent": "InstaLILY-Hackathon/1.0"})
            with urllib.request.urlopen(req, timeout=60) as response:
                total = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 1024 * 1024  # 1MB chunks

                with open(dest, "wb") as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            mb_done = downloaded / (1024 * 1024)
                            mb_total = total / (1024 * 1024)
                            print(f"\r  {mb_done:.0f}/{mb_total:.0f} MB ({pct:.1f}%)", end="", flush=True)

                print()  # newline after progress

                # Validate downloaded size
                if total > 0 and downloaded != total:
                    logger.warning(f"Size mismatch: expected {total}, got {downloaded}")
                    if attempt < retries:
                        continue
                    return False

            logger.info(f"Saved to {dest} ({downloaded / (1024*1024):.1f} MB)")
            return True

        except Exception as e:
            logger.warning(f"Download failed (attempt {attempt}): {e}")
            if dest.exists():
                dest.unlink()
            if attempt >= retries:
                return False

    return False


def check_url_reachable(url: str) -> tuple[bool, int]:
    """Check if a URL is reachable via HEAD request. Returns (reachable, content_length)."""
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "InstaLILY-Hackathon/1.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            size = int(response.headers.get("Content-Length", 0))
            return True, size
    except Exception as e:
        logger.warning(f"URL check failed for {url}: {e}")
        return False, 0


def download_speaker_videos(speaker: str, output_dir: Path = RAW_VIDEO_DIR) -> Path | None:
    """
    Download and extract video archive for one speaker.
    Returns the directory containing .mpg files, or None on failure.
    """
    if speaker in UNAVAILABLE_SPEAKERS:
        logger.warning(f"Speaker {speaker} is not available in the GRID corpus")
        return None

    speaker_dir = output_dir / speaker
    if speaker_dir.is_dir() and any(speaker_dir.glob("*.mpg")):
        logger.info(f"Speaker {speaker} videos already exist at {speaker_dir}, skipping download")
        return speaker_dir

    zip_url = f"{ZENODO_BASE_URL}/{speaker}.zip"
    zip_dest = output_dir / f"{speaker}.zip"

    if not download_file(zip_url, zip_dest):
        logger.error(f"Failed to download {speaker}.zip")
        return None

    # Extract
    logger.info(f"Extracting {zip_dest}...")
    try:
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(output_dir)
    except zipfile.BadZipFile:
        logger.error(f"Corrupt zip file: {zip_dest}")
        zip_dest.unlink(missing_ok=True)
        return None

    # Clean up zip
    zip_dest.unlink(missing_ok=True)

    # Find the extracted directory (may be nested: s1/video/s1/*.mpg or s1/*.mpg)
    if speaker_dir.is_dir():
        mpg_files = list(speaker_dir.rglob("*.mpg"))
        if mpg_files:
            # If files are in a subdirectory, flatten
            first_mpg_parent = mpg_files[0].parent
            if first_mpg_parent != speaker_dir:
                for mpg in mpg_files:
                    shutil.move(str(mpg), str(speaker_dir / mpg.name))
                # Clean up empty subdirs
                for d in sorted(speaker_dir.rglob("*"), reverse=True):
                    if d.is_dir() and not list(d.iterdir()):
                        d.rmdir()

            logger.info(f"Extracted {len(list(speaker_dir.glob('*.mpg')))} videos for {speaker}")
            return speaker_dir

    logger.error(f"Could not find extracted videos for {speaker}")
    return None


def download_alignments(output_dir: Path = ALIGNMENT_DIR) -> Path | None:
    """
    Download and extract word-level alignment files.
    Returns the alignments root directory, or None on failure.
    """
    # Check if already extracted
    if output_dir.is_dir() and any(output_dir.rglob("*.align")):
        logger.info(f"Alignments already exist at {output_dir}, skipping download")
        return output_dir

    zip_url = f"{ZENODO_BASE_URL}/alignments.zip"
    zip_dest = DATA_DIR / "alignments.zip"

    if not download_file(zip_url, zip_dest):
        logger.error("Failed to download alignments.zip")
        return None

    logger.info(f"Extracting alignments...")
    try:
        with zipfile.ZipFile(zip_dest, "r") as zf:
            zf.extractall(DATA_DIR)
    except zipfile.BadZipFile:
        logger.error(f"Corrupt zip file: {zip_dest}")
        zip_dest.unlink(missing_ok=True)
        return None

    zip_dest.unlink(missing_ok=True)

    # The zip extracts to alignments/ directory. Move to expected location if needed.
    extracted_dir = DATA_DIR / "alignments"
    if extracted_dir.is_dir() and extracted_dir != output_dir:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(extracted_dir), str(output_dir))

    align_count = len(list(output_dir.rglob("*.align")))
    logger.info(f"Extracted {align_count} alignment files")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Download GRID corpus subset from Zenodo")
    parser.add_argument("--speakers", nargs="+", default=DEFAULT_SPEAKERS)
    parser.add_argument("--output-dir", type=Path, default=RAW_VIDEO_DIR)
    parser.add_argument("--alignments-dir", type=Path, default=ALIGNMENT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Only check URLs, don't download")
    parser.add_argument("--local-path", type=Path, default=None, help="Copy from local path instead of downloading")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.dry_run:
        print("=== Dry run: checking URL reachability ===")
        # Check alignment URL
        url = f"{ZENODO_BASE_URL}/alignments.zip"
        ok, size = check_url_reachable(url)
        print(f"  alignments.zip: {'OK' if ok else 'FAILED'} ({size / (1024*1024):.1f} MB)")

        for speaker in args.speakers:
            if speaker in UNAVAILABLE_SPEAKERS:
                print(f"  {speaker}.zip: UNAVAILABLE (speaker not in corpus)")
                continue
            url = f"{ZENODO_BASE_URL}/{speaker}.zip"
            ok, size = check_url_reachable(url)
            print(f"  {speaker}.zip: {'OK' if ok else 'FAILED'} ({size / (1024*1024):.1f} MB)")
        return

    if args.local_path:
        print(f"=== Copying from local path: {args.local_path} ===")
        # Copy videos
        for speaker in args.speakers:
            src = args.local_path / speaker
            if src.is_dir():
                dst = args.output_dir / speaker
                dst.mkdir(parents=True, exist_ok=True)
                for mpg in src.glob("*.mpg"):
                    shutil.copy2(str(mpg), str(dst / mpg.name))
                print(f"  Copied {speaker}: {len(list(dst.glob('*.mpg')))} videos")
        # Copy alignments
        align_src = args.local_path / "alignments"
        if align_src.is_dir():
            if args.alignments_dir.exists():
                shutil.rmtree(args.alignments_dir)
            shutil.copytree(str(align_src), str(args.alignments_dir))
            print(f"  Copied alignments: {len(list(args.alignments_dir.rglob('*.align')))} files")
        return

    # Download alignments first (smaller, needed by all speakers)
    print("=== Downloading GRID corpus ===")
    download_alignments(args.alignments_dir)

    # Download speaker videos
    for speaker in args.speakers:
        download_speaker_videos(speaker, args.output_dir)

    print("=== Download complete ===")


if __name__ == "__main__":
    main()
