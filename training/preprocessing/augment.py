"""
augment.py

Usage:
    python augment.py --src <crops_dir>

Arguments:
    --src   Path to directory containing 224x224 face crop .jpg files
            Augmentation is applied in-place.
"""

import os
import argparse
import random
import subprocess
import tempfile
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def apply_ffmpeg_compression(image_path: str, crf: int) -> bool:
    """
    Apply H.264 compression to a single image via FFmpeg.
    Encodes the image as a single-frame video and decodes it back.
    Returns True on success, False on failure.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        tmp_video_path = tmp_video.name

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_out:
        tmp_out_path = tmp_out.name

    try:
        # Step 1: Encode image as single-frame H.264 video
        encode_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            image_path,
            "-vcodec",
            "libx264",
            "-crf",
            str(crf),
            "-preset",
            "fast",
            tmp_video_path,
        ]
        result = subprocess.run(
            encode_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            return False

        # Step 2: Decode the compressed video back to a single frame
        decode_cmd = [
            "ffmpeg",
            "-y",
            "-i",
            tmp_video_path,
            "-frames:v",
            "1",
            tmp_out_path,
        ]
        result = subprocess.run(
            decode_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            return False

        # Step 3: Overwrite the original file
        import shutil

        shutil.move(tmp_out_path, image_path)
        return True

    except Exception as e:
        print(f"[WARNING] FFmpeg error on {image_path}: {e}")
        return False
    finally:
        for path in [tmp_video_path, tmp_out_path]:
            if os.path.exists(path):
                os.remove(path)


def augment_dataset(src_dir: str) -> None:
    image_files = [
        f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    total = len(image_files)
    augmented = 0
    skipped = 0
    failed = 0

    for filename in image_files:
        img_path = os.path.join(src_dir, filename)

        # Apply augmentation with probability AUG_PROBABILITY
        if random.random() > config.AUG_PROBABILITY:
            skipped += 1
            continue

        crf = random.randint(config.AUG_CRF_MIN, config.AUG_CRF_MAX)
        success = apply_ffmpeg_compression(img_path, crf)

        if success:
            augmented += 1
        else:
            print(f"[WARNING] Augmentation failed for: {img_path}")
            failed += 1

    print(f"\n[DONE] Total images: {total}")
    print(f"[DONE] Augmented (compressed): {augmented}")
    print(f"[DONE] Skipped (probability): {skipped}")
    print(f"[DONE] Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply FFmpeg compression augmentation to face crop images."
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Directory of face crop images to augment in-place",
    )
    args = parser.parse_args()

    augment_dataset(args.src)
