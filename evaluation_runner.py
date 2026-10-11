"""
evaluation_runner.py

Offline batch evaluation of the deepfake detection pipeline against a
labelled folder of video clips and/or images (real vs. deepfake), including
non-FaceForensics++ sources such as diffusion-model-generated samples.
Runs each sample through the same face_detector -> preprocessor -> inference
pipeline as the live app, without the screen-capture/GUI loop.

Usage:
    python evaluation_runner.py [dataset_root] [--real-subdir real]
        [--fake-subdir deepfake] [--frames-per-video N] [--csv-path path]
        [--no-db]

    dataset_root defaults to test_samples/ (populated by prepare_ff_samples.py).

Expected folder structure:
    <dataset_root>/
        real/       — .jpg/.jpeg/.png images and/or .mp4/.avi/.mov/.mkv clips
        deepfake/   — .jpg/.jpeg/.png images and/or .mp4/.avi/.mov/.mkv clips

At least MIN_SAMPLES files in total, and at least 1 per class, are required.

Results are written both to a CSV file and to the local session-events
database with ground_truth_label populated (event_type='CLASSIFICATION'),
so runtime.storage.database.compute_classification_metrics() and
get_score_distribution() can analyse them afterward. The run is bracketed
by SESSION start/stop events, so it also appears as a session in the app's
history table; pass --no-db to skip all database writes. The BitMind API's
1 request/second rate limit is enforced automatically (inside the shared
DeepfakeInferenceEngine).
"""

import argparse
import csv
import os
import sys
from datetime import datetime

import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import config
from runtime.detection.face_detector import FaceDetector
from runtime.detection.preprocessor import Preprocessor
from runtime.detection.inference import DeepfakeInferenceEngine, BitmindAPIError
from runtime.storage.database import log_event

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
CSV_FIELDS = ["filename", "label", "predicted", "p_fake", "api_ms", "correct"]
MIN_SAMPLES = 10
DEFAULT_DATASET_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_samples")


def _iter_dataset(root: str, subdir: str) -> list:
    """Return sorted (path, is_video) tuples for supported image/video files in root/subdir."""
    folder = os.path.join(root, subdir)
    if not os.path.isdir(folder):
        return []
    items = []
    for f in sorted(os.listdir(folder)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTS:
            items.append((os.path.join(folder, f), False))
        elif ext in VIDEO_EXTS:
            items.append((os.path.join(folder, f), True))
    return items


def _iter_video_frames(path: str, n: int):
    """Yield up to n evenly-spaced BGR frames from a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"    [SKIP] Cannot open: {os.path.basename(path)}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return
    step = max(1, total // n)
    for i in range(n):
        pos = min(i * step, total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def _classify_sample(frame, detector: FaceDetector, preprocessor: Preprocessor,
                      engine: DeepfakeInferenceEngine, sample_id: str) -> tuple:
    """
    Run one frame through the full pipeline; pick the worst (highest P_fake)
    face when multiple are detected. Falls back to submitting the whole
    frame if no face is detected (common for pre-cropped datasets).

    Returns:
        (FaceResult, faces_detected: int). Raises BitmindAPIError on failure.
    """
    face_crops = detector.detect(frame)

    crops = [crop for crop, _ in face_crops] if face_crops else [frame]
    best_result = None
    for i, crop in enumerate(crops):
        jpeg = preprocessor.process(crop)
        result = engine.run_single(jpeg, bbox={}, face_id=f"{sample_id}_{i}")
        if best_result is None or result.p_fake_raw > best_result.p_fake_raw:
            best_result = result
    return best_result, len(face_crops)


def main():
    parser = argparse.ArgumentParser(description="Offline batch evaluation of the deepfake detection pipeline.")
    parser.add_argument("dataset_root", nargs="?", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--real-subdir", default="real")
    parser.add_argument("--fake-subdir", default="deepfake")
    parser.add_argument("--frames-per-video", type=int, default=config.EVAL_FRAMES_PER_VIDEO)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--no-db", action="store_true",
                        help="Do not write SESSION/CLASSIFICATION events to the session database.")
    args = parser.parse_args()
    write_db = not args.no_db

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        print(f"[ERROR] Not a directory: {dataset_root}")
        sys.exit(1)

    real_items = _iter_dataset(dataset_root, args.real_subdir)
    fake_items = _iter_dataset(dataset_root, args.fake_subdir)
    total = len(real_items) + len(fake_items)
    if total < MIN_SAMPLES:
        print(f"[ERROR] Minimum {MIN_SAMPLES} files required ({total} found). Add images/videos to:")
        print(f"    {os.path.join(dataset_root, args.real_subdir)}")
        print(f"    {os.path.join(dataset_root, args.fake_subdir)}")
        sys.exit(1)
    if not real_items or not fake_items:
        print("[ERROR] Both the real and deepfake folders must contain at least 1 file.")
        sys.exit(1)

    detector = FaceDetector()
    preprocessor = Preprocessor()
    try:
        engine = DeepfakeInferenceEngine()
    except EnvironmentError as e:
        print(f"[ERROR] {e}")
        detector.close()
        sys.exit(1)

    csv_path = args.csv_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )

    print("=" * 62)
    print("  OFFLINE BATCH EVALUATION")
    print("=" * 62)
    print(f"  Dataset : {dataset_root}")
    print(f"  Samples : {len(real_items)} real, {len(fake_items)} deepfake ({total} files)")
    print(f"  CSV out : {csv_path}")
    print(f"  DB log  : {'yes' if write_db else 'no (--no-db)'}")
    print(f"  Threshold: {config.FAKE_THRESHOLD * 100:.0f}%  (P_fake >= threshold → DEEPFAKE)")
    print()
    print(f"  {'#':<5} {'Label':<9} {'Sample':<34} {'P_fake':>7}  {'Decision':<9} {'OK?':<4}  API ms")
    print(f"  {'-'*5} {'-'*9} {'-'*34} {'-'*7}  {'-'*9} {'-'*4}  {'-'*6}")

    if write_db:
        log_event("SESSION", "Detection started.")

    rows = []
    errors = 0
    idx = 1
    try:
        for label, items in [("REAL", real_items), ("DEEPFAKE", fake_items)]:
            for path, is_video in items:
                fname = os.path.basename(path)
                frames = list(_iter_video_frames(path, args.frames_per_video)) if is_video else [cv2.imread(path)]

                for frame_idx, frame in enumerate(frames):
                    if frame is None:
                        print(f"  {idx:<5} {label:<9} {fname:<34}  ERROR: cannot load frame")
                        errors += 1
                        idx += 1
                        continue

                    sample_id = f"{fname}_{frame_idx}"
                    display_name = sample_id if len(sample_id) <= 34 else sample_id[:31] + "..."
                    try:
                        result, n_faces = _classify_sample(frame, detector, preprocessor, engine, sample_id)
                    except BitmindAPIError as e:
                        print(f"  {idx:<5} {label:<9} {display_name:<34}  ERROR: {e}")
                        if write_db:
                            log_event("ERROR", f"Evaluation sample failed: {sample_id}: {e}")
                        errors += 1
                        idx += 1
                        continue

                    correct = result.risk_level == label
                    if write_db:
                        log_event(
                            "CLASSIFICATION", detail=f"sample={sample_id} risk={result.risk_level}",
                            peak_confidence=result.p_fake_raw, api_response_ms=result.api_response_ms,
                            ground_truth_label=label,
                        )
                    rows.append({
                        "filename": sample_id, "label": label, "predicted": result.risk_level,
                        "p_fake": result.p_fake_raw, "api_ms": result.api_response_ms, "correct": correct,
                    })
                    ok_mark = "OK" if correct else "FAIL"
                    nf_note = " (no face)" if n_faces == 0 else ""
                    print(f"  {idx:<5} {label:<9} {display_name:<34} {result.p_fake_raw*100:>6.1f}%  "
                          f"{result.risk_level:<9} {ok_mark:<4}  {result.api_response_ms:>5.0f}ms{nf_note}")
                    idx += 1
    finally:
        if write_db:
            log_event("SESSION", "Detection stopped.")
        detector.close()
        engine.close()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print_summary(rows, errors)
    print(f"  CSV written to : {csv_path}")
    if write_db:
        print("  Run runtime.storage.database.compute_classification_metrics() for a full")
        print("  precision/recall/F1 breakdown, or get_score_distribution() for the raw scores.")
    print("=" * 62)


def _pct(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100 if denominator else 0.0


def _mean(values: list) -> float:
    return sum(values) / len(values) if values else 0.0


def print_summary(rows: list, errors: int):
    """Print overall and per-class accuracy, mean P_fake per class, and mean API response time."""
    real_rows = [r for r in rows if r["label"] == "REAL"]
    fake_rows = [r for r in rows if r["label"] == "DEEPFAKE"]
    real_correct = sum(r["correct"] for r in real_rows)
    fake_correct = sum(r["correct"] for r in fake_rows)
    n = len(rows)

    print()
    print("=" * 62)
    print("  RESULTS")
    print("=" * 62)
    print(f"  Samples evaluated : {n}  ({errors} skipped due to load/API errors)")
    print()
    print("  Correctly identified")
    print(f"  {'─'*44}")
    print(f"  Real faces      : {real_correct}/{len(real_rows)}   ({_pct(real_correct, len(real_rows)):.1f}%)")
    print(f"  Deepfake faces  : {fake_correct}/{len(fake_rows)}   ({_pct(fake_correct, len(fake_rows)):.1f}%)")
    print(f"  {'─'*44}")
    print(f"  Overall accuracy : {real_correct + fake_correct}/{n}   ({_pct(real_correct + fake_correct, n):.1f}%)")
    print()
    print("  Average confidence score (P_fake)")
    print(f"    Real  faces : {_mean([r['p_fake'] for r in real_rows]) * 100:>6.1f}%")
    print(f"    Deepfake    : {_mean([r['p_fake'] for r in fake_rows]) * 100:>6.1f}%")
    print()
    print(f"  Mean API response : {_mean([r['api_ms'] for r in rows]):>6.0f} ms")


if __name__ == "__main__":
    main()
