"""
evaluation_runner.py

Offline batch evaluation of the deepfake detection pipeline against labelled
folders of video clips and/or images: one real folder and one or more
deepfake folders, including non-FaceForensics++ sources such as
diffusion-model-generated samples. Runs each sample through the same
face_detector -> preprocessor -> inference pipeline as the live app, without
the screen-capture/GUI loop. Each sample is scored by exactly one BitMind
call on its largest face, so the metrics measure BitMind's accuracy on the
face crops the app sends.

Usage:
    python evaluation_runner.py [dataset_root] [--real-subdir real]
        [--fake-subdir deepfake [more ...]] [--limit N] [--frames-per-video N]
        [--csv-path path] [--no-db]
    python evaluation_runner.py --summary-only --csv-path path

    dataset_root defaults to test_samples/.

Expected folder structure:
    <dataset_root>/
        real/       — .jpg/.jpeg/.png images and/or .mp4/.avi/.mov/.mkv clips
        deepfake/   — .jpg/.jpeg/.png images and/or .mp4/.avi/.mov/.mkv clips

A FaceForensics++ root is evaluated straight from its videos by naming its
subset folders. The real set is classified once and shared by every method,
e.g. the first 100 videos of each folder, one frame (the midpoint) per video:
    python evaluation_runner.py <ff_root> --real-subdir original
        --fake-subdir Deepfakes FaceSwap Face2Face FaceShifter NeuralTextures
        --limit 100 --frames-per-video 1 --csv-path ff_accuracy_results.csv
For the standard nested FF++ layout, pass the full subpaths instead
(e.g. original_sequences/youtube/c23/videos).

At least MIN_SAMPLES files in total, and at least 1 per folder, are required.

Each result is appended to the CSV as soon as it arrives. Re-running with
the same --csv-path skips samples already in it, so an interrupted run
resumes and failed samples are retried. The summary reports, per deepfake
folder, TP/FN, recall, precision, accuracy, F1 and ROC-AUC against the shared
real set, plus a threshold sweep for recalibrating FAKE_THRESHOLD;
--summary-only re-prints it from the CSV without any API calls, recomputing
decisions from P_fake with the current FAKE_THRESHOLD.

Unless --no-db is passed, results are also written to the local
session-events database with ground_truth_label populated
(event_type='CLASSIFICATION'), bracketed by SESSION start/stop events, so the
run also appears as a session in the app's history table. The BitMind API's
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
import config
from runtime.detection.face_detector import FaceDetector
from runtime.detection.preprocessor import Preprocessor
from runtime.detection.inference import DeepfakeInferenceEngine, BitmindAPIError
from runtime.storage.database import log_event

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
CSV_FIELDS = ["source", "filename", "label", "p_fake", "predicted", "correct", "api_ms", "faces"]
MIN_SAMPLES = 10
SWEEP_THRESHOLDS = [t / 10 for t in range(1, 10)]
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
    """Yield up to n BGR frames, each from the middle of one of n equal segments (n=1 gives the midpoint)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"    [SKIP] Cannot open: {os.path.basename(path)}")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / n))
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
    cap.release()


def _iter_samples(groups: list, frames_per_video: int):
    """Yield (label, source, sample_id, frame) for every image and sampled video frame; frame is None if unreadable."""
    for label, source, items in groups:
        for path, is_video in items:
            fname = os.path.basename(path)
            frames = _iter_video_frames(path, frames_per_video) if is_video else [cv2.imread(path)]
            for frame_idx, frame in enumerate(frames):
                yield label, source, f"{fname}_{frame_idx}", frame


def _classify_sample(frame, detector: FaceDetector, preprocessor: Preprocessor,
                      engine: DeepfakeInferenceEngine, sample_id: str) -> tuple:
    """Classify one frame with a single BitMind call; returns (FaceResult, faces_detected). Raises BitmindAPIError."""
    faces = detector.detect(frame)
    # Only the largest face, so each sample's score is exactly one BitMind verdict on the crop the live
    # app would send; scoring extra faces (often detector false positives) and keeping the max would mix
    # detector errors into BitMind's accuracy. The whole frame is sent when no face is found (pre-cropped datasets).
    crop = faces[0][0] if faces else frame
    return engine.run_single(preprocessor.process(crop), bbox={}, face_id=sample_id), len(faces)

<<<<<<< HEAD
    Returns:
        (FaceResult, faces_detected: int). Raises BitmindAPIError on failure.
    """
    face_crops = detector.detect(frame)
=======
>>>>>>> 3bef4a09fe8c51c26d29703668d4889ad5fe4d31

def _load_rows(csv_path: str) -> list:
    """Return the result rows already in csv_path (none if it does not exist), with numeric fields parsed."""
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [
            {**r, "p_fake": float(r["p_fake"]), "api_ms": float(r["api_ms"]), "faces": int(r["faces"])}
            for r in csv.DictReader(f)
        ]


def main():
    parser = argparse.ArgumentParser(description="Offline batch evaluation of the deepfake detection pipeline.")
    parser.add_argument("dataset_root", nargs="?", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--real-subdir", default="real")
    parser.add_argument("--fake-subdir", nargs="+", default=["deepfake"],
                        help="One or more deepfake folders, e.g. FF++ methods; all share the real set.")
    parser.add_argument("--limit", type=int, default=None, help="Max files per folder (first N in name order).")
    parser.add_argument("--frames-per-video", type=int, default=config.EVAL_FRAMES_PER_VIDEO)
    parser.add_argument("--csv-path", default=None,
                        help="Results CSV; samples already in it are skipped, so re-running resumes.")
    parser.add_argument("--no-db", action="store_true",
                        help="Do not write SESSION/CLASSIFICATION events to the session database.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Skip evaluation; summarise the existing --csv-path.")
    args = parser.parse_args()

    if args.summary_only:
        if not args.csv_path or not os.path.exists(args.csv_path):
            parser.error("--summary-only needs an existing --csv-path")
        print_summary(_load_rows(args.csv_path), errors=0)
        return
    write_db = not args.no_db

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.isdir(dataset_root):
        print(f"[ERROR] Not a directory: {dataset_root}")
        sys.exit(1)

    groups = []
    for label, subdir in [("REAL", args.real_subdir)] + [("DEEPFAKE", s) for s in args.fake_subdir]:
        items = _iter_dataset(dataset_root, subdir)[:args.limit]
        if not items:
            print(f"[ERROR] No images/videos found in {os.path.join(dataset_root, subdir)}")
            sys.exit(1)
        groups.append((label, subdir, items))
    total = sum(len(items) for _, _, items in groups)
    if total < MIN_SAMPLES:
        print(f"[ERROR] Minimum {MIN_SAMPLES} files required ({total} found).")
        sys.exit(1)
    expected = sum(args.frames_per_video if is_video else 1 for _, _, items in groups for _, is_video in items)

    csv_path = args.csv_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    done = {(r["source"], r["filename"]) for r in _load_rows(csv_path)}

    detector = FaceDetector()
    preprocessor = Preprocessor()
    try:
        engine = DeepfakeInferenceEngine()
    except EnvironmentError as e:
        print(f"[ERROR] {e}")
        detector.close()
        sys.exit(1)

    print("=" * 70)
    print("  OFFLINE BATCH EVALUATION")
    print("=" * 70)
    print(f"  Dataset : {dataset_root}")
    print(f"  Samples : {', '.join(f'{len(items)} {source}' for _, source, items in groups)} ({expected} frames)")
    print(f"  CSV out : {csv_path}" + (f"  (resuming: {len(done)} already done)" if done else ""))
    print(f"  DB log  : {'yes' if write_db else 'no (--no-db)'}")
    print(f"  Threshold: {config.FAKE_THRESHOLD * 100:.0f}%  (P_fake >= threshold → DEEPFAKE)")
    print()
    print(f"  {'#':<11} {'Source':<14} {'Sample':<24} {'P_fake':>7}  {'Decision':<9} {'OK?':<4}  API ms")
    print(f"  {'-'*11} {'-'*14} {'-'*24} {'-'*7}  {'-'*9} {'-'*4}  {'-'*6}")

    if write_db:
        log_event("SESSION", "Detection started.")

    errors = 0
    write_header = not os.path.exists(csv_path)
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            samples = _iter_samples(groups, args.frames_per_video)
            for idx, (label, source, sample_id, frame) in enumerate(samples, start=1):
                if (source, sample_id) in done:
                    continue
                prefix = f"  {f'{idx}/{expected}':<11} {source[:14]:<14} {sample_id[:24]:<24}"
                if frame is None:
                    print(f"{prefix}  ERROR: cannot load frame")
                    errors += 1
                    continue
                try:
                    result, n_faces = _classify_sample(frame, detector, preprocessor, engine, sample_id)
                except BitmindAPIError as e:
                    print(f"{prefix}  ERROR: {e}")
                    if write_db:
                        log_event("ERROR", f"Evaluation sample failed: {source}/{sample_id}: {e}")
                    errors += 1
                    continue

                correct = result.risk_level == label
                if write_db:
                    log_event(
                        "CLASSIFICATION", detail=f"sample={source}/{sample_id} risk={result.risk_level}",
                        peak_confidence=result.p_fake_raw, api_response_ms=result.api_response_ms,
                        ground_truth_label=label,
                    )
                writer.writerow({
                    "source": source, "filename": sample_id, "label": label,
                    "p_fake": round(result.p_fake_raw, 4), "predicted": result.risk_level, "correct": correct,
                    "api_ms": round(result.api_response_ms), "faces": n_faces,
                })
                f.flush()
                nf_note = " (no face)" if n_faces == 0 else ""
                print(f"{prefix} {result.p_fake_raw*100:>6.1f}%  {result.risk_level:<9} "
                      f"{'OK' if correct else 'FAIL':<4}  {result.api_response_ms:>5.0f}ms{nf_note}")
    finally:
        if write_db:
            log_event("SESSION", "Detection stopped.")
        detector.close()
        engine.close()

    print_summary(_load_rows(csv_path), errors)
    print(f"  CSV written to : {csv_path}")
    print("=" * 70)


def _rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(values: list) -> float:
    return _rate(sum(values), len(values))


def _auc(real: list, fake: list) -> float:
    """ROC-AUC: chance that a random deepfake scores a higher P_fake than a random real face (ties count half)."""
    return _rate(sum((f > r) + 0.5 * (f == r) for f in fake for r in real), len(fake) * len(real))


def print_summary(rows: list, errors: int):
    """Print real-face results, per-source metrics against the shared real set, and a FAKE_THRESHOLD sweep."""
    threshold = config.FAKE_THRESHOLD
    real = [r["p_fake"] for r in rows if r["label"] == "REAL"]
    sources: dict = {}
    for r in rows:
        if r["label"] == "DEEPFAKE":
            sources.setdefault(r["source"], []).append(r["p_fake"])
    if len(sources) > 1:
        sources["All"] = [p for scores in list(sources.values()) for p in scores]
    false_alarms = sum(p >= threshold for p in real)

    print()
    print("=" * 70)
    print(f"  RESULTS  (DEEPFAKE if P_fake >= {threshold:.2f})")
    print("=" * 70)
    print(f"  Samples evaluated : {len(rows)}  ({errors} skipped due to load/API errors this run)")
    print(f"  No face detected  : {sum(r['faces'] == 0 for r in rows)}  (whole frame submitted)")
    print(f"  Mean API response : {_mean([r['api_ms'] for r in rows]):.0f} ms")
    print(f"  Real faces        : {len(real)}, correctly REAL {len(real) - false_alarms} "
          f"({_rate(len(real) - false_alarms, len(real)) * 100:.1f}%), mean P_fake {_mean(real) * 100:.1f}%")
    print()
    print(f"  {'Deepfake source':<16}{'n':>5}{'TP':>5}{'FN':>5}{'Recall':>8}{'Prec.':>8}{'Acc.':>8}{'F1':>8}"
          f"{'AUC':>7}{'P_fake':>8}")
    print(f"  {'-' * 78}")
    for name, fake in sources.items():
        tp = sum(p >= threshold for p in fake)
        recall = _rate(tp, len(fake))
        precision = _rate(tp, tp + false_alarms)
        accuracy = _rate(tp + len(real) - false_alarms, len(fake) + len(real))
        f1 = _rate(2 * precision * recall, precision + recall)
        print(f"  {name[:16]:<16}{len(fake):>5}{tp:>5}{len(fake) - tp:>5}{recall * 100:>7.1f}%{precision * 100:>7.1f}%"
              f"{accuracy * 100:>7.1f}%{f1 * 100:>7.1f}%{_auc(real, fake):>7.3f}{_mean(fake) * 100:>7.1f}%")

    print()
    print("  THRESHOLD SWEEP  (FPR = real faces flagged as DEEPFAKE; recall per deepfake source)")
    print(f"  {'Threshold':<11}{'FPR':>7}" + "".join(f"{name[:11]:>12}" for name in sources))
    for t in sorted(set(SWEEP_THRESHOLDS) | {threshold}):
        fpr = _rate(sum(p >= t for p in real), len(real))
        recalls = "".join(f"{_rate(sum(p >= t for p in fake), len(fake)) * 100:>11.1f}%" for fake in sources.values())
        print(f"  {t:<11.2f}{fpr * 100:>6.1f}%{recalls}" + ("  (current)" if t == threshold else ""))


if __name__ == "__main__":
    main()
