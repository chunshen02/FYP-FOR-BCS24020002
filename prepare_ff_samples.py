"""
prepare_ff_samples.py

Extracts labelled frames from a FaceForensics++ dataset and saves them to
test_samples/real/ and test_samples/deepfake/ for use with evaluation_runner.py.

Supports both the standard FF++ layout and the flat Kaggle repack:

  Flat (Kaggle):                Standard:
  <root>/original/*.mp4         <root>/original_sequences/youtube/c23/videos/*.mp4
  <root>/Deepfakes/*.mp4        <root>/manipulated_sequences/Deepfakes/c23/videos/*.mp4

Usage:
    python prepare_ff_samples.py <ff_root> [options]

    ff_root   : FaceForensics++ root directory
    --real N  : total real frames to save     (default 10)
    --fake N  : total fake frames to save     (default 10)
    --method  : Deepfakes | Face2Face | FaceSwap | NeuralTextures | FaceShifter
                (default Deepfakes)
    --quality : c0 | c23 | c40  — only needed for standard layout (default c23)

Example:
    python prepare_ff_samples.py "C:/FaceForensics++_C23" --real 15 --fake 15
"""

import os
import sys
import argparse
import cv2

METHODS  = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures",
            "FaceShifter", "DeepFakeDetection"]
QUALITIES = ["c0", "c23", "c40"]
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_samples")


def _find_videos(directory: str) -> list[str]:
    """Return sorted list of .mp4 paths inside directory (flat search, one level)."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".mp4")
    )


def _resolve_dirs(ff_root: str, method: str, quality: str) -> tuple[str, str]:
    """
    Return (real_dir, fake_dir) by probing both the flat Kaggle layout and the
    standard FF++ nested layout.
    """
    # Flat Kaggle layout: <root>/original/  and  <root>/<Method>/
    flat_real = os.path.join(ff_root, "original")
    flat_fake = os.path.join(ff_root, method)
    if os.path.isdir(flat_real) and _find_videos(flat_real):
        return flat_real, flat_fake

    # Standard FF++ layout
    std_real = os.path.join(ff_root, "original_sequences", "youtube", quality, "videos")
    std_fake = os.path.join(ff_root, "manipulated_sequences", method, quality, "videos")
    return std_real, std_fake


def _extract_from_video(video_path: str, n: int, out_dir: str, prefix: str) -> int:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [SKIP] Cannot open: {os.path.basename(video_path)}")
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return 0
    step = max(1, total // n)
    saved = 0
    for i in range(n):
        pos = min(i * step, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            break
        path = os.path.join(out_dir, f"{prefix}_{i:04d}.jpg")
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
    cap.release()
    return saved


def _collect(video_dir: str, n_target: int, out_dir: str, tag: str) -> int:
    videos = _find_videos(video_dir)
    if not videos:
        print(f"  [ERROR] No .mp4 files in: {video_dir}")
        return 0
    os.makedirs(out_dir, exist_ok=True)
    per_video = max(1, -(-n_target // len(videos)))  # ceiling division
    saved = 0
    for i, vpath in enumerate(videos):
        if saved >= n_target:
            break
        want = min(per_video, n_target - saved)
        n = _extract_from_video(vpath, want, out_dir, f"{tag}_{i:03d}")
        saved += n
        print(f"  [{tag.upper():4}] {os.path.basename(vpath):<30} → {n} frame(s)")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ff_root")
    parser.add_argument("--real",    type=int, default=10)
    parser.add_argument("--fake",    type=int, default=10)
    parser.add_argument("--method",  default="Deepfakes", choices=METHODS)
    parser.add_argument("--quality", default="c23",       choices=QUALITIES)
    args = parser.parse_args()

    ff_root = os.path.abspath(args.ff_root)
    if not os.path.isdir(ff_root):
        print(f"[ERROR] Not a directory: {ff_root}")
        sys.exit(1)

    real_dir, fake_dir = _resolve_dirs(ff_root, args.method, args.quality)

    print("[PREPARE] FaceForensics++ → test_samples/")
    print(f"  Root   : {ff_root}")
    print(f"  Method : {args.method}  |  Target: {args.real} real + {args.fake} fake")
    print(f"  Real   : {real_dir}")
    print(f"  Fake   : {fake_dir}")
    print()

    real_out = os.path.join(SAMPLES_DIR, "real")
    fake_out = os.path.join(SAMPLES_DIR, "deepfake")

    r = _collect(real_dir, args.real, real_out, "real")
    f = _collect(fake_dir, args.fake, fake_out, "fake")

    print()
    print(f"[DONE]  {r} real   → {real_out}")
    print(f"        {f} fake   → {fake_out}")
    print(f"        Total : {r + f} samples")

    if r + f < 10:
        print(f"\n[WARNING] Only {r + f} collected — minimum 10 required for evaluation_runner.py.")
    else:
        print("\nRun:  python evaluation_runner.py")


if __name__ == "__main__":
    main()
