"""
accuracy_test.py

Evaluates deepfake detection accuracy against a labelled image dataset.
Sends each image through the full pipeline: face detection → JPEG encode → BitMind API.

Usage:
    python accuracy_test.py

Expected folder structure:
    test_samples/
        real/       — .jpg/.jpeg/.png images confirmed to be genuine faces
        deepfake/   — .jpg/.jpeg/.png images confirmed to be AI-generated faces

Minimum 10 images total across both folders, at least 1 per class.
If a face is not detected in an image, the whole image is submitted as-is.
The API rate limit (1 req/sec) is enforced automatically; allow ~2s per sample.
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

import config
from runtime.detection.face_detector import FaceDetector
from runtime.detection.preprocessor import Preprocessor
from runtime.detection.inference import DeepfakeInferenceEngine

MIN_SAMPLES = 10
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}
SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_samples")


def _image_paths(label: str) -> list[str]:
    folder = os.path.join(SAMPLES_DIR, label)
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    )


def _process(path: str, detector: FaceDetector,
             preprocessor: Preprocessor,
             engine: DeepfakeInferenceEngine) -> dict:
    frame = cv2.imread(path)
    if frame is None:
        return {"error": f"Cannot load {os.path.basename(path)}"}

    h, w = frame.shape[:2]
    scale = config.DETECTION_UPSCALE
    scaled = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    face_crops = detector.detect(scaled)

    if face_crops:
        # Downscale crops back to original resolution proportions, then pick worst face
        best_result = None
        for i, (crop, _) in enumerate(face_crops):
            jpeg = preprocessor.process(crop)
            r = engine.run_single(jpeg, bbox={}, face_id=f"{os.path.basename(path)}_{i}")
            if best_result is None or r.p_fake_raw > best_result.p_fake_raw:
                best_result = r
        return {
            "faces": len(face_crops),
            "p_fake": best_result.p_fake_raw,
            "decision": best_result.risk_level,
            "api_ms": best_result.api_response_ms,
        }

    # No face detected — submit full image as crop (common for pre-cropped datasets)
    jpeg = preprocessor.process(frame)
    r = engine.run_single(jpeg, bbox={}, face_id=os.path.basename(path))
    return {
        "faces": 0,
        "p_fake": r.p_fake_raw,
        "decision": r.risk_level,
        "api_ms": r.api_response_ms,
    }


def main():
    real_paths = _image_paths("real")
    fake_paths = _image_paths("deepfake")
    total = len(real_paths) + len(fake_paths)

    print("=" * 62)
    print("  DEEPFAKE DETECTION ACCURACY TEST")
    print("=" * 62)
    print(f"  Samples  : {len(real_paths)} real, {len(fake_paths)} deepfake ({total} total)")
    print(f"  Threshold: {config.FAKE_THRESHOLD * 100:.0f}%  (P_fake >= threshold → DEEPFAKE)")
    print()

    if total < MIN_SAMPLES:
        print(f"[ERROR] Minimum {MIN_SAMPLES} samples required ({total} found).")
        print(f"  Add .jpg/.jpeg/.png images to:")
        print(f"    {os.path.join(SAMPLES_DIR, 'real')}")
        print(f"    {os.path.join(SAMPLES_DIR, 'deepfake')}")
        return

    if not real_paths or not fake_paths:
        print("[ERROR] Both real/ and deepfake/ must contain at least 1 image.")
        return

    detector = FaceDetector()
    preprocessor = Preprocessor()

    try:
        engine = DeepfakeInferenceEngine()
    except EnvironmentError as e:
        print(f"[ERROR] {e}")
        return

    print(f"  Estimated time: ~{int(total * 2)}s  (API rate limit: 1 req/sec)\n")
    print(f"  {'#':<4} {'Label':<9} {'Filename':<34} {'P_fake':>7}  {'Decision':<9} {'OK?':<4}  API ms")
    print(f"  {'-'*4} {'-'*9} {'-'*34} {'-'*7}  {'-'*9} {'-'*4}  {'-'*6}")

    results = []
    idx = 1

    for label, paths in [("REAL", real_paths), ("DEEPFAKE", fake_paths)]:
        for path in paths:
            fname = os.path.basename(path)
            display_name = fname if len(fname) <= 34 else fname[:31] + "..."
            print(f"  {idx:<4} {label:<9} {display_name:<34}", end="", flush=True)

            r = _process(path, detector, preprocessor, engine)

            if "error" in r:
                print(f"  ERROR: {r['error']}")
                results.append({"label": label, "error": r["error"]})
                idx += 1
                continue

            correct = r["decision"] == label
            ok_mark = "OK" if correct else "FAIL"
            nf_note = " (no face)" if r["faces"] == 0 else ""
            print(f"  {r['p_fake']*100:>6.1f}%  {r['decision']:<9} {ok_mark:<4}  {r['api_ms']:>5.0f}ms{nf_note}")

            results.append({
                "label": label,
                "p_fake": r["p_fake"],
                "decision": r["decision"],
                "faces": r["faces"],
                "api_ms": r["api_ms"],
                "correct": correct,
            })
            idx += 1

    detector.close()
    engine.close()

    # ── Results ──────────────────────────────────────────────────────────
    valid = [r for r in results if "error" not in r]
    errors = len(results) - len(valid)
    n = len(valid)

    real_valid = [r for r in valid if r["label"] == "REAL"]
    fake_valid = [r for r in valid if r["label"] == "DEEPFAKE"]
    real_correct = sum(1 for r in real_valid if r["correct"])
    fake_correct = sum(1 for r in fake_valid if r["correct"])
    total_correct = real_correct + fake_correct

    accuracy = total_correct / n if n > 0 else 0.0
    real_acc = real_correct / len(real_valid) if real_valid else 0.0
    fake_acc = fake_correct / len(fake_valid) if fake_valid else 0.0

    avg_p_fake_real = sum(r["p_fake"] for r in real_valid) / len(real_valid) if real_valid else 0.0
    avg_p_fake_fake = sum(r["p_fake"] for r in fake_valid) / len(fake_valid) if fake_valid else 0.0
    avg_api_ms = sum(r["api_ms"] for r in valid) / n if n > 0 else 0.0

    print()
    print("=" * 62)
    print("  RESULTS")
    print("=" * 62)
    print(f"  Samples tested : {n}  ({errors} skipped due to load errors)")
    print()
    print(f"  Correctly identified")
    print(f"  {'─'*44}")
    print(f"  Real faces      : {real_correct}/{len(real_valid)}   ({real_acc*100:.1f}%)")
    print(f"  Deepfake faces  : {fake_correct}/{len(fake_valid)}   ({fake_acc*100:.1f}%)")
    print(f"  {'─'*44}")
    print(f"  Overall Accuracy : {total_correct}/{n}   ({accuracy*100:.1f}%)")
    print()
    print(f"  Average confidence score (P_fake)")
    print(f"    Real  faces : {avg_p_fake_real*100:>6.1f}%")
    print(f"    Deepfake    : {avg_p_fake_fake*100:>6.1f}%")
    print()
    print(f"  Mean API response  : {avg_api_ms:>6.0f} ms")
    print("=" * 62)


if __name__ == "__main__":
    main()
