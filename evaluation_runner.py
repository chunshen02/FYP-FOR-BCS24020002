"""
evaluation_runner.py

Offline batch evaluation of the deepfake detection pipeline against a
labelled folder of video clips and/or images (real vs. deepfake), including
non-FaceForensics++ sources such as diffusion-model-generated samples.
Runs each sample through the same face_detector -> preprocessor -> inference
pipeline as the live app, without the screen-capture/GUI loop.

Usage:
    python evaluation_runner.py <dataset_root> [--real-subdir real]
        [--fake-subdir deepfake] [--frames-per-video N] [--csv-path path]

Expected folder structure:
    <dataset_root>/
        real/       — .jpg/.jpeg/.png images and/or .mp4/.avi/.mov/.mkv clips
        deepfake/   — .jpg/.jpeg/.png images and/or .mp4/.avi/.mov/.mkv clips

Results are written both to a CSV file and to the local session-events
database with ground_truth_label populated (event_type='CLASSIFICATION'),
so runtime.storage.database.compute_classification_metrics() and
get_score_distribution() can analyse them afterward. The BitMind API's
1 request/second rate limit is enforced automatically (inside the shared
DeepfakeInferenceEngine), same as accuracy_test.py.
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
    h, w = frame.shape[:2]
    scale = config.DETECTION_UPSCALE
    scaled = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    face_crops = detector.detect(scaled)

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
    parser.add_argument("dataset_root")
    parser.add_argument("--real-subdir", default="real")
    parser.add_argument("--fake-subdir", default="deepfake")
    parser.add_argument("--frames-per-video", type=int, default=config.EVAL_FRAMES_PER_VIDEO)
    parser.add_argument("--csv-path", default=None)
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        print(f"[ERROR] Not a directory: {dataset_root}")
        sys.exit(1)

    real_items = _iter_dataset(dataset_root, args.real_subdir)
    fake_items = _iter_dataset(dataset_root, args.fake_subdir)
    total = len(real_items) + len(fake_items)
    if total == 0:
        print(f"[ERROR] No supported image/video files found under {dataset_root}")
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
    print()
    print(f"  {'#':<5} {'Label':<9} {'Sample':<34} {'P_fake':>7}  {'Decision':<9} {'OK?':<4}  API ms")
    print(f"  {'-'*5} {'-'*9} {'-'*34} {'-'*7}  {'-'*9} {'-'*4}  {'-'*6}")

    log_event("SESSION", "Detection started.")

    rows = []
    idx = 1
    try:
        for label, items in [("REAL", real_items), ("DEEPFAKE", fake_items)]:
            for path, is_video in items:
                fname = os.path.basename(path)
                frames = list(_iter_video_frames(path, args.frames_per_video)) if is_video else [cv2.imread(path)]

                for frame_idx, frame in enumerate(frames):
                    if frame is None:
                        print(f"  {idx:<5} {label:<9} {fname:<34}  ERROR: cannot load frame")
                        idx += 1
                        continue

                    sample_id = f"{fname}_{frame_idx}"
                    display_name = sample_id if len(sample_id) <= 34 else sample_id[:31] + "..."
                    try:
                        result, _n_faces = _classify_sample(frame, detector, preprocessor, engine, sample_id)
                    except BitmindAPIError as e:
                        print(f"  {idx:<5} {label:<9} {display_name:<34}  ERROR: {e}")
                        log_event("ERROR", f"Evaluation sample failed: {sample_id}: {e}")
                        idx += 1
                        continue

                    correct = result.risk_level == label
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
                    print(f"  {idx:<5} {label:<9} {display_name:<34} {result.p_fake_raw*100:>6.1f}%  "
                          f"{result.risk_level:<9} {ok_mark:<4}  {result.api_response_ms:>5.0f}ms")
                    idx += 1
    finally:
        log_event("SESSION", "Detection stopped.")
        detector.close()
        engine.close()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    correct_n = sum(1 for r in rows if r["correct"])
    print()
    print("=" * 62)
    print(f"  Samples evaluated : {n}")
    print(f"  Overall accuracy  : {correct_n}/{n}  ({(correct_n / n * 100) if n else 0.0:.1f}%)")
    print(f"  CSV written to    : {csv_path}")
    print("  Run runtime.storage.database.compute_classification_metrics() for a full")
    print("  precision/recall/F1 breakdown, or get_score_distribution() for the raw scores.")
    print("=" * 62)


if __name__ == "__main__":
    main()
