"""Detection accuracy of the app's BitMind pipeline on FaceForensics++ (c23) videos."""

import os

# Silence MediaPipe / TFLite start-up logs so the progress lines stay readable (same as main.py).
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import argparse
import csv
import statistics
import sys
import time

import cv2

import config
from realtime_performance import _hold_test_lock, _run_info
from runtime.detection.face_detector import FaceDetector
from runtime.detection.preprocessor import Preprocessor
from runtime.detection.inference import (
    DeepfakeInferenceEngine, BitmindAPIError, BitmindAuthError, BitmindRateLimitError, verdict_decided,
)

CSV_FIELDS = ["subset", "video", "label", "p_fakes", "calls", "mean_api_ms"]
DEFAULT_CSV = os.path.join(config.RESULTS_DIR, "accuracy_results.csv")
TRANSIENT_RETRY_DELAYS_SEC = (2.0, 4.0, 8.0)
SWEEP_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def _subsets() -> list:
    return [(config.FF_REAL_SUBSET, "REAL")] + [(s, "DEEPFAKE") for s in config.FF_FAKE_SUBSETS]


def _list_videos(dataset: str, limit: int) -> list:
    """Return (subset, label, path) for the first `limit` videos (by name) of every subset."""
    videos = []
    for subset, label in _subsets():
        folder = os.path.join(dataset, subset)
        if not os.path.isdir(folder):
            sys.exit(f"[ERROR] Subset folder not found: {folder}")
        names = sorted(f for f in os.listdir(folder) if f.lower().endswith(".mp4"))[:limit]
        videos += [(subset, label, os.path.join(folder, name)) for name in names]
    return videos


def _sample_frames(path: str) -> list:
    """Return the frames at the middles of CONFIRM_VERDICTS equal segments, so each extra verdict sees a fresh frame."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(config.CONFIRM_VERDICTS if total > 0 else 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((i + 0.5) * total / config.CONFIRM_VERDICTS))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def _classify(engine: DeepfakeInferenceEngine, jpeg_bytes: bytes) -> tuple:
    """Return (P_fake, API ms) for one crop, waiting out rate limits and retrying transient failures."""
    rate_limit_pause = config.BITMIND_RATE_LIMIT_BACKOFF_SEC
    retry_delays = list(TRANSIENT_RETRY_DELAYS_SEC)
    while True:
        try:
            result = engine.run_single_attempt(jpeg_bytes, bbox={}, face_id="")
            return result.p_fake_raw, result.api_response_ms
        except BitmindRateLimitError as e:
            pause = e.retry_after or rate_limit_pause
            rate_limit_pause = min(rate_limit_pause * 2, config.BITMIND_RATE_LIMIT_BACKOFF_MAX_SEC)
            print(f"    Rate limited by BitMind — waiting {pause:.0f}s")
            time.sleep(pause)
        except BitmindAuthError:
            raise
        except BitmindAPIError:
            if not retry_delays:
                raise
            time.sleep(retry_delays.pop(0))


def _evaluate_video(path: str, detector: FaceDetector, preprocessor: Preprocessor,
                    engine: DeepfakeInferenceEngine) -> tuple:
    """Classify a video's most confident face with the app's confirmation rule; return (votes, API ms list)."""
    votes, api_ms = [], []
    for frame in _sample_frames(path):
        faces = detector.detect(frame)
        if not faces:
            continue
        jpeg_bytes, _ = preprocessor.process_with_array(faces[0][0])
        p_fake, ms = _classify(engine, jpeg_bytes)
        votes.append(p_fake)
        api_ms.append(ms)
        if verdict_decided(votes):
            break
    return votes, api_ms


def _confirmed(votes: list) -> float:
    """Median of the verdicts the app would use: the shortest prefix that settles the face, else all of them."""
    for n in range(1, len(votes) + 1):
        if verdict_decided(votes[:n]):
            return statistics.median(votes[:n])
    return statistics.median(votes)


def _load_rows(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [
            {**r, "votes": [float(p) for p in r["p_fakes"].split(";") if p],
             "calls": int(r["calls"]), "mean_api_ms": float(r["mean_api_ms"])}
            for r in csv.DictReader(f)
        ]


def _run(dataset: str, limit: int, csv_path: str):
    done = {(r["subset"], r["video"]) for r in _load_rows(csv_path)}
    videos = [v for v in _list_videos(dataset, limit) if (v[0], os.path.basename(v[2])) not in done]
    print(f"[ACCURACY] {len(videos)} videos to evaluate ({len(done)} already in {csv_path}).")
    print(f"  Threshold {config.FAKE_THRESHOLD:.0%}; a first DEEPFAKE verdict is confirmed with up to "
          f"{config.CONFIRM_VERDICTS} verdicts ({config.CONFIRM_FAKE_VOTES} DEEPFAKE needed).")
    if not videos:
        return
    try:
        engine = DeepfakeInferenceEngine()
    except EnvironmentError as e:
        sys.exit(f"[ERROR] {e}")
    detector = FaceDetector()
    preprocessor = Preprocessor()

    write_header = not os.path.exists(csv_path)
    t_start = time.perf_counter()
    active_key = engine.active_key
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if write_header:
                writer.writeheader()
            for i, (subset, label, path) in enumerate(videos, start=1):
                if engine.active_key != active_key:
                    print(f"  BitMind key {active_key} rejected or used up — switched to {engine.active_key}")
                    active_key = engine.active_key
                video = os.path.basename(path)
                prefix = f"  [{i:>{len(str(len(videos)))}}/{len(videos)}] {subset + '/' + video:<34}"
                try:
                    votes, api_ms = _evaluate_video(path, detector, preprocessor, engine)
                except BitmindAuthError as e:
                    print("[ERROR] BitMind rejected the API key — check BITMIND_API_KEY / "
                          f"BITMIND_API_KEY_SECONDARY in .env. ({e})")
                    break
                except BitmindAPIError as e:
                    print(f"{prefix} SKIPPED ({e}); it is retried on the next run")
                    continue
                writer.writerow({
                    "subset": subset, "video": video, "label": label,
                    "p_fakes": ";".join(f"{p:.4f}" for p in votes), "calls": len(votes),
                    "mean_api_ms": round(statistics.mean(api_ms)) if api_ms else 0,
                })
                f.flush()

                eta_min = (time.perf_counter() - t_start) / i * (len(videos) - i) / 60
                if votes:
                    predicted = "DEEPFAKE" if _confirmed(votes) >= config.FAKE_THRESHOLD else "REAL"
                    scores = " ".join(f"{p:.0%}" for p in votes)
                    outcome = f"{scores:<14} {predicted:<9} {'OK' if predicted == label else 'WRONG':<5}"
                else:
                    outcome = f"{'no face found':<30}"
                print(f"{prefix} {outcome} ETA {eta_min:.0f} min", flush=True)
    except KeyboardInterrupt:
        print("\n  Interrupted — run the same command again to resume.")
    finally:
        detector.close()
        engine.close()


def _rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _auc(real: list, fake: list) -> float:
    """ROC-AUC: chance that a random deepfake scores a higher P_fake than a random real face (ties count half)."""
    return _rate(sum((f > r) + 0.5 * (f == r) for f in fake for r in real), len(fake) * len(real))


def _metrics(real: list, fake: list, threshold: float) -> dict:
    tp = sum(p >= threshold for p in fake)
    fp = sum(p >= threshold for p in real)
    tn, fn = len(real) - fp, len(fake) - tp
    precision, recall = _rate(tp, tp + fp), _rate(tp, len(fake))
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": _rate(tp + tn, len(real) + len(fake)),
        "precision": precision, "recall": recall, "fpr": _rate(fp, len(real)),
        "f1": _rate(2 * precision * recall, precision + recall),
    }


def _decision_row(name: str, real: list, fake: list, threshold: float) -> str:
    m = _metrics(real, fake, threshold)
    balanced = (m["recall"] + (1 - m["fpr"])) / 2
    return (f"  {name:<22}{m['tp']:>5}{m['fp']:>5}{m['tn']:>5}{m['fn']:>5}{m['accuracy']:>8.1%}{balanced:>9.1%}"
            f"{m['precision']:>8.1%}{m['recall']:>8.1%}{m['f1']:>7.1%}{m['fpr']:>7.1%}")


def _report_lines(real: list, methods: list, threshold: float) -> list:
    """Per-method and combined tables in the report's layout, from (method, deepfake scores) pairs."""
    lines = [
        f"  Per method (confirmed) — each row = the method's deepfakes + the same {len(real)} real videos; "
        "Mean = average over methods",
        f"  {'Method':<18}{'TP':>5}{'TN':>5}{'FP':>5}{'FN':>5}{'Total':>7}{'Accuracy':>10}{'Precision':>11}"
        f"{'Recall':>8}{'F1':>8}",
        "  " + "-" * 82,
    ]
    per_method = [_metrics(real, fake, threshold) for _, fake in methods]
    for (method, fake), m in zip(methods, per_method):
        lines.append(f"  {method:<18}{m['tp']:>5}{m['tn']:>5}{m['fp']:>5}{m['fn']:>5}{len(fake) + len(real):>7}"
                     f"{m['accuracy']:>10.1%}{m['precision']:>11.1%}{m['recall']:>8.1%}{m['f1']:>8.1%}")
    mean = {k: statistics.mean(m[k] for m in per_method) for k in ("accuracy", "precision", "recall", "f1")}
    lines.append(f"  {'Mean':<18}{'—':>5}{'—':>5}{'—':>5}{'—':>5}{'—':>7}{mean['accuracy']:>10.1%}"
                 f"{mean['precision']:>11.1%}{mean['recall']:>8.1%}{mean['f1']:>8.1%}")

    fake = [p for _, scores in methods for p in scores]
    c = _metrics(real, fake, threshold)
    lines += [
        "",
        f"  Combined (confirmed) — all {len(methods)} methods ({len(real)} real + {len(fake)} deepfake)",
        f"  {'Metric':<24}Result",
        "  " + "-" * 82,
        f"  {'True Positives (TP)':<24}{c['tp']} of {len(fake)} deepfakes correctly flagged",
        f"  {'True Negatives (TN)':<24}{c['tn']} of {len(real)} real faces correctly cleared",
        f"  {'False Positives (FP)':<24}{c['fp']} real faces wrongly flagged",
        f"  {'False Negatives (FN)':<24}{c['fn']} deepfakes missed",
        f"  {'Accuracy':<24}{c['accuracy']:.1%} ({c['tp'] + c['tn']}/{len(real) + len(fake)} correct)",
        f"  {'Precision':<24}{c['precision']:.1%} ({c['tp']}/{c['tp'] + c['fp']} flagged faces were deepfakes)",
        f"  {'Recall':<24}{c['recall']:.1%} ({c['tp']}/{len(fake)} deepfakes detected)",
        f"  {'F1 Score':<24}{c['f1']:.1%}",
    ]
    return lines


def _summary_lines(rows: list) -> list:
    threshold = config.FAKE_THRESHOLD
    scored = [{**r, "first": r["votes"][0], "final": _confirmed(r["votes"])} for r in rows if r["votes"]]

    def scores(key: str, label: str, subset: str = None) -> list:
        return [r[key] for r in scored if r["label"] == label and subset in (None, r["subset"])]

    def correct_rate(values: list, label: str) -> float:
        return _rate(sum((p >= threshold) == (label == "DEEPFAKE") for p in values), len(values))

    real_first, fake_first = scores("first", "REAL"), scores("first", "DEEPFAKE")
    calls = sum(r["calls"] for r in rows)
    api_mean = _rate(sum(r["mean_api_ms"] * r["calls"] for r in rows), calls)
    line = "=" * 92
    lines = [
        line,
        "  DETECTION ACCURACY — FaceForensics++ c23",
        *_run_info(),
        "  (The results CSV is resumable, so its rows may come from earlier runs on other code.)",
        f"  Threshold {threshold:.0%}. Single = first BitMind verdict only. Confirmed = app rule: a first REAL",
        f"  verdict is final, otherwise {config.CONFIRM_FAKE_VOTES} of {config.CONFIRM_VERDICTS} verdicts "
        f"decide and the score is their median.",
        f"  Videos: {len(scored)} scored ({len(real_first)} real, {len(fake_first)} deepfake), "
        f"{len(rows) - len(scored)} with no detectable face (excluded).",
        f"  BitMind calls: {calls} ({_rate(calls, len(scored)):.2f} per video), mean response {api_mean:.0f} ms.",
        line,
        f"  {'Decision':<22}{'TP':>5}{'FP':>5}{'TN':>5}{'FN':>5}{'Acc.':>8}{'Bal.Acc':>9}"
        f"{'Prec.':>8}{'Recall':>8}{'F1':>7}{'FPR':>7}",
        "  " + "-" * 88,
        _decision_row("Single verdict", real_first, fake_first, threshold),
        _decision_row("Confirmed (app)", scores("final", "REAL"), scores("final", "DEEPFAKE"), threshold),
        "",
        "  Per subset — correct = deepfake caught / real video kept REAL; AUC (first verdict) vs the real videos",
        f"  {'Subset':<18}{'n':>5}{'Correct single':>16}{'Correct confirmed':>19}{'AUC':>8}{'Mean P_fake':>13}",
        "  " + "-" * 79,
    ]
    for subset, label in _subsets():
        sub_first, sub_final = scores("first", label, subset), scores("final", label, subset)
        if not sub_first:
            lines.append(f"  {subset:<18}{0:>5}  (not evaluated yet)")
            continue
        auc =f"{_auc(real_first, sub_first):.3f}" if label == "DEEPFAKE" else ""
        lines.append(
            f"  {subset:<18}{len(sub_first):>5}{correct_rate(sub_first, label):>16.1%}"
            f"{correct_rate(sub_final, label):>19.1%}{auc:>8}{_rate(sum(sub_first), len(sub_first)):>13.1%}"
        )

    methods = [(subset, scores("final", label, subset)) for subset, label in _subsets() if label == "DEEPFAKE"]
    methods = [(subset, fake) for subset, fake in methods if fake]
    if methods:
        lines += ["", *_report_lines(scores("final", "REAL"), methods, threshold)]

    lines += [
        "",
        "  Threshold sweep (single verdict) — for choosing FAKE_THRESHOLD",
        f"  {'Threshold':<12}{'FPR':>8}{'Recall':>9}{'Bal.Acc':>10}",
    ]
    for t in sorted(set(SWEEP_THRESHOLDS) | {threshold}):
        fpr = _rate(sum(p >= t for p in real_first), len(real_first))
        recall = _rate(sum(p >= t for p in fake_first), len(fake_first))
        current = "  (current)" if t == threshold else ""
        lines.append(f"  {t:<12.2f}{fpr:>8.1%}{recall:>9.1%}{(recall + 1 - fpr) / 2:>10.1%}{current}")
    lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser(description="Detection accuracy of the app's BitMind pipeline on FaceForensics++ (c23).")
    parser.add_argument("--dataset", default=config.FF_DATASET_ROOT, help="FaceForensics++_C23 folder.")
    parser.add_argument("--limit", type=int, default=config.ACCURACY_VIDEOS_PER_SUBSET,
                        help="Videos per subset (first N by name).")
    parser.add_argument("--csv", default=DEFAULT_CSV,
                        help="Results CSV; videos already in it are skipped, so re-running resumes.")
    parser.add_argument("--summary-only", action="store_true",
                        help="Re-print the summary from the CSV without any API calls.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    if not args.summary_only:
        lock = _hold_test_lock()  # referenced until main() returns, so the lock file stays open and locked
        _run(args.dataset, args.limit, args.csv)

    rows = _load_rows(args.csv)
    if not rows:
        sys.exit(f"[ERROR] No results in {args.csv}.")
    lines = _summary_lines(rows)
    summary_path = os.path.join(os.path.dirname(os.path.abspath(args.csv)), "accuracy_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"  Results CSV : {args.csv}\n  Summary     : {summary_path}")


if __name__ == "__main__":
    main()
