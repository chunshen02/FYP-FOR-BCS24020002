"""Real-time performance of the app's detection pipeline, fed with FaceForensics++ (c23) videos."""

import os

# Silence MediaPipe / TFLite start-up logs so the progress lines stay readable (same as main.py).
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import argparse
import csv
import msvcrt
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import psutil

import config
from runtime.capture.screen_capture import ScreenCapture
from runtime.detection.face_detector import FaceDetector
from runtime.detection.preprocessor import Preprocessor
from runtime.detection.inference import (
    DeepfakeInferenceEngine, BitmindAPIError, BitmindAuthError, BitmindRateLimitError,
    risk_level_for, verdict_decided,
)
from runtime.threads.detection_worker import _FaceTracker

_FRAME_INTERVAL = 1.0 / config.DISPLAY_FPS
# Longer than the tracker's grace period, so every clip's face is tracked as a
# new face instead of inheriting the previous clip's locked verdict.
GAP_SECONDS = 1.0
VIDEO_CSV_FIELDS = ["clip", "subset", "video", "label", "verdict", "correct", "calls",
                    "first_verdict_ms", "final_verdict_ms", "api_time_ms", "app_latency_ms", "faces_tracked"]


def _playlist(dataset: str, per_subset: int) -> list:
    """Return (subset, label, path), cycling through the real subset and each fake method."""
    subsets = [(config.FF_REAL_SUBSET, "REAL")] + [(s, "DEEPFAKE") for s in config.FF_FAKE_SUBSETS]
    names = {}
    for subset, _ in subsets:
        folder = os.path.join(dataset, subset)
        if not os.path.isdir(folder):
            sys.exit(f"[ERROR] Subset folder not found: {folder}")
        names[subset] = sorted(f for f in os.listdir(folder) if f.lower().endswith(".mp4"))[:per_subset]
    return [
        (subset, label, os.path.join(dataset, subset, names[subset][i]))
        for i in range(per_subset) for subset, label in subsets if i < len(names[subset])
    ]


def _letterbox(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale a frame to fit width x height and centre it on black, like full-screen playback."""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - new_w) // 2, (height - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return canvas


class _VideoScreen:
    """Plays the playlist in real time at screen size, with a black gap between clips."""

    def __init__(self, playlist: list, width: int, height: int):
        self._playlist = playlist
        self._width, self._height = width, height
        self._black = np.zeros((height, width, 3), dtype=np.uint8)
        self._clip = -1
        self._cap = None
        self._gap_until = 0.0
        self._fps = 25.0
        self._clip_start = 0.0
        self._next_index = 0      # index of the next frame cap.grab() will return
        self._current = self._black

    def frame(self, now: float) -> tuple:
        """Return (screen frame, clip index or None during a gap), or (None, None) once every clip has played."""
        if self._cap is None:
            if now < self._gap_until:
                return self._black, None
            self._clip += 1
            if self._clip >= len(self._playlist):
                return None, None
            subset, _, path = self._playlist[self._clip]
            print(f"  ▶ [{self._clip + 1}/{len(self._playlist)}] {subset}/{os.path.basename(path)}", flush=True)
            self._cap = cv2.VideoCapture(path)
            self._fps = self._cap.get(cv2.CAP_PROP_FPS) or self._fps
            self._clip_start = now
            self._next_index = 0
            self._current = self._black

        # Real-time playback: skip straight to the frame due now, decoding only that one fully.
        due = int((now - self._clip_start) * self._fps)
        advanced = False
        while self._next_index <= due:
            if not self._cap.grab():
                self._cap.release()
                self._cap = None
                self._gap_until = now + GAP_SECONDS
                return self._black, None
            self._next_index += 1
            advanced = True
        if advanced:
            ok, raw = self._cap.retrieve()
            if ok:
                self._current = _letterbox(raw, self._width, self._height)
        return self._current, self._clip

    def close(self):
        if self._cap is not None:
            self._cap.release()


class _Pipeline:
    """Queue and verdict state shared by the main loop and the API thread, with DetectionWorker's rules."""

    def __init__(self):
        self.lock = threading.Lock()
        self.pending: dict = {}   # face_id -> (crop, bbox), round-robin order
        self.locked: set = set()
        self.votes: dict = {}     # face_id -> P_fake verdicts until verdict_decided
        self.accepted: set = set()
        self.inflight: str | None = None
        self.faces: dict = {}     # face_id -> record for the per-clip results
        self.api_ms: list = []
        self.failures = 0
        self.rate_limited = 0
        self.auth_error: str | None = None


def _api_loop(engine: DeepfakeInferenceEngine, state: _Pipeline, stop: threading.Event):
    """Classify queued faces round-robin, confirming DEEPFAKE verdicts, as DetectionWorker._api_loop does."""
    preprocessor = Preprocessor()
    rate_limit_pause = config.BITMIND_RATE_LIMIT_BACKOFF_SEC
    while not stop.is_set():
        with state.lock:
            if state.pending:
                face_id, (crop, bbox) = next(iter(state.pending.items()))
                del state.pending[face_id]
                state.pending[face_id] = (crop, bbox)
                state.inflight = face_id
            else:
                face_id = None
        if face_id is None:
            time.sleep(0.05)
            continue

        jpeg_bytes, _ = preprocessor.process_with_array(crop)
        t_call = time.perf_counter()
        try:
            result = engine.run_single_attempt(jpeg_bytes, bbox, face_id)
        except BitmindAuthError as e:
            state.auth_error = str(e)
            stop.set()
            return
        except BitmindRateLimitError as e:
            pause = e.retry_after or rate_limit_pause
            rate_limit_pause = min(rate_limit_pause * 2, config.BITMIND_RATE_LIMIT_BACKOFF_MAX_SEC)
            state.rate_limited += 1
            print(f"    Rate limited by BitMind — pausing {pause:.0f}s", flush=True)
            stop.wait(pause)
            continue
        except BitmindAPIError:
            state.failures += 1
            continue
        finally:
            # Failed and rate-limited requests count as BitMind time too: the app did not cause that wait.
            with state.lock:
                state.inflight = None
                state.faces[face_id]["api_s"] += time.perf_counter() - t_call
        rate_limit_pause = config.BITMIND_RATE_LIMIT_BACKOFF_SEC

        now = time.perf_counter()
        with state.lock:
            state.api_ms.append(result.api_response_ms)
            votes = state.votes.setdefault(face_id, [])
            votes.append(result.p_fake_raw)
            record = state.faces[face_id]
            record["calls"] += 1
            record["first_verdict"] = record["first_verdict"] or now
            if not verdict_decided(votes):
                continue
            del state.votes[face_id]
            state.locked.add(face_id)
            state.pending.pop(face_id, None)
            record["final_verdict"] = now
            record["verdict"] = risk_level_for(statistics.median(votes))
        print(f"    {face_id}: {record['verdict']} after {now - record['first_seen']:.1f}s "
              f"({record['calls']} call{'s' if record['calls'] > 1 else ''})", flush=True)


def _run(playlist: list) -> dict:
    """Play the playlist through the pipeline and return the raw measurements."""
    try:
        engine = DeepfakeInferenceEngine()
    except EnvironmentError as e:
        sys.exit(f"[ERROR] {e}")
    capture = ScreenCapture()
    capture.start()
    screen_h, screen_w = capture.read().shape[:2]
    screen = _VideoScreen(playlist, screen_w, screen_h)
    detector = FaceDetector()
    tracker = _FaceTracker()
    state = _Pipeline()
    stop = threading.Event()
    api_thread = threading.Thread(target=_api_loop, args=(engine, state, stop), daemon=True)
    api_thread.start()

    process = psutil.Process()
    cores = psutil.cpu_count()
    stages = {"capture": [], "detect": [], "track": [], "total": []}
    fps_samples, cpu_samples, ram_samples = [], [], []
    run_start = time.perf_counter()
    measuring = False
    window_start = window_frames = window_cpu = window_player_cpu = 0

    print(f"  Screen {screen_w}x{screen_h}; first {config.PERF_WARMUP_SECONDS}s are warm-up (not measured).")
    try:
        while not stop.is_set():
            tick_start = time.perf_counter()
            # The simulated video player's decode + scaling is not app work: it is
            # excluded from the latency figures and subtracted from the CPU figure.
            player_cpu_start = time.thread_time()
            frame, clip = screen.frame(tick_start)
            player_cpu = time.thread_time() - player_cpu_start
            if frame is None:
                break

            t0 = time.perf_counter()
            capture.read()  # a real screen grab every tick, as in the live app; the image itself is unused
            t_captured = time.perf_counter()
            faces = detector.detect(frame)
            t_detected = time.perf_counter()
            with state.lock:
                protected = frozenset({state.inflight}) if state.inflight else frozenset()
            face_ids = tracker.update([b for _, b, _ in faces], protected_ids=protected)
            alive = tracker.alive_ids()
            keep = alive | protected
            with state.lock:
                state.pending = {k: v for k, v in state.pending.items() if k in keep}
                state.votes = {k: v for k, v in state.votes.items() if k in keep}
                state.locked &= alive
                state.accepted &= alive
                for (crop, bbox, score), fid in zip(faces, face_ids):
                    if fid not in state.faces:
                        state.faces[fid] = {"clip": clip, "first_seen": t0, "first_verdict": None, "final_verdict": None,
                                            "verdict": "", "calls": 0, "api_s": 0.0, "frames": 0}
                    state.faces[fid]["frames"] += 1
                    if score >= config.FACE_ACCEPT_SCORE:
                        state.accepted.add(fid)
                    if fid in state.locked or fid not in state.accepted:
                        continue
                    if fid in state.pending or len(state.pending) < config.MAX_PENDING_FACES:
                        state.pending[fid] = (crop, bbox)
            t_end = time.perf_counter()

            if not measuring and t_end - run_start >= config.PERF_WARMUP_SECONDS:
                measuring = True
                window_start, window_frames, window_player_cpu = t_end, 0, 0.0
                window_cpu = sum(process.cpu_times()[:2])
            if measuring:
                stages["capture"].append((t_captured - t0) * 1000)
                stages["detect"].append((t_detected - t_captured) * 1000)
                stages["track"].append((t_end - t_detected) * 1000)
                stages["total"].append((t_end - t0) * 1000)
                window_frames += 1
                window_player_cpu += player_cpu
                if t_end - window_start >= 1.0:
                    cpu_now = sum(process.cpu_times()[:2])
                    elapsed = t_end - window_start
                    fps_samples.append(window_frames / elapsed)
                    cpu_samples.append(100 * (cpu_now - window_cpu - window_player_cpu) / (elapsed * cores))
                    ram_samples.append(process.memory_info().rss / 2 ** 20)
                    window_start, window_frames, window_player_cpu, window_cpu = t_end, 0, 0.0, cpu_now

            # Same tick-rate cap as the live app, so CPU reflects real usage rather than a busy loop.
            sleep_time = _FRAME_INTERVAL - (time.perf_counter() - tick_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n  Interrupted — summarising what was measured so far.")
    finally:
        stop.set()
        api_thread.join(timeout=config.BITMIND_API_TIMEOUT_SEC + 1)
        screen.close()
        detector.close()
        engine.close()
        capture.stop()

    return {
        "screen": (screen_w, screen_h), "stages": stages, "fps": fps_samples, "cpu": cpu_samples,
        "ram": ram_samples, "faces": state.faces, "api_ms": state.api_ms, "failures": state.failures,
        "rate_limited": state.rate_limited, "auth_error": state.auth_error,
    }


def _clip_rows(playlist: list, faces: dict) -> list:
    """One row per clip, for its main face: the track seen in most frames, so a stray non-face box never stands in for it."""
    rows = []
    for clip, (subset, label, path) in enumerate(playlist):
        clip_faces = [r for r in faces.values() if r["clip"] == clip]
        main = max(clip_faces, key=lambda r: r["frames"], default=None)

        def ms(key: str) -> str:
            return f"{(main[key] - main['first_seen']) * 1000:.0f}" if main and main[key] else ""

        verdict = main["verdict"] if main else ""
        decided = bool(main and main["final_verdict"])
        rows.append({
            "clip": clip + 1, "subset": subset, "video": os.path.basename(path), "label": label,
            "verdict": verdict, "correct": verdict == label if verdict else "", "calls": main["calls"] if main else 0,
            "first_verdict_ms": ms("first_verdict"), "final_verdict_ms": ms("final_verdict"),
            "api_time_ms": f"{main['api_s'] * 1000:.0f}" if decided else "",
            "app_latency_ms": f"{(main['final_verdict'] - main['first_seen'] - main['api_s']) * 1000:.0f}"
                              if decided else "",
            "faces_tracked": len(clip_faces),
        })
    return rows


def _mean(values: list) -> float | None:
    return statistics.mean(values) if values else None


def _p95(values: list) -> float | None:
    return float(np.percentile(values, 95)) if values else None


def _hold_test_lock():
    """
    Exit if another test script is already running, else hold the lock until this process ends.

    Concurrent tests compete for the screen, the CPU and BitMind's per-key
    rate limit, which skews both runs' figures. The OS releases the lock when
    the process exits, even on a crash or Ctrl+C, so it can never go stale.
    Call it from main(), not at import: screen_face_pipeline's player child
    process re-imports the main module and must not lock itself out.
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    handle = open(os.path.join(config.RESULTS_DIR, ".test.lock"), "a")
    try:
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        sys.exit("[ERROR] Another test script is already running — wait for it to finish "
                 "(tests compete for the screen, CPU and BitMind's rate limit).")
    return handle


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=config.BASE_DIR, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run_info() -> list:
    """Summary lines naming the code version and detection settings, so result files from different versions can be told apart."""
    commit = _git("rev-parse", "--short", "HEAD")
    if commit is None:
        code = "unknown (not a git checkout)"
    # results/ is excluded: changed or deleted test outputs are not a code change.
    elif _git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results"):
        code = f"{commit} + uncommitted changes"
    else:
        code = commit
    return [
        f"  Code: {code} (summary written {datetime.now():%Y-%m-%d %H:%M})",
        f"  Settings: DETECTION_TILE_GRID={config.DETECTION_TILE_GRID}, FACE_ACCEPT_SCORE={config.FACE_ACCEPT_SCORE}, "
        f"FAKE_THRESHOLD={config.FAKE_THRESHOLD:.2f}",
    ]


def _summary_lines(stats: dict, rows: list) -> list:
    stages = stats["stages"]
    final_ms = [float(r["final_verdict_ms"]) for r in rows if r["final_verdict_ms"]]
    first_ms = [float(r["first_verdict_ms"]) for r in rows if r["first_verdict_ms"]]
    api_time_ms = [float(r["api_time_ms"]) for r in rows if r["api_time_ms"]]
    app_ms = [float(r["app_latency_ms"]) for r in rows if r["app_latency_ms"]]
    mean_fps, min_fps = _mean(stats["fps"]), min(stats["fps"], default=None)
    mean_latency, mean_final, mean_app = _mean(stages["total"]), _mean(final_ms), _mean(app_ms)
    mean_cpu, peak_ram = _mean(stats["cpu"]), max(stats["ram"], default=None)

    def fmt(value: float | None, unit: str) -> str:
        return "n/a" if value is None else f"{value:.1f}{unit}"

    def row(label: str, value: float | None, unit: str, target: str, passed: bool | None) -> str:
        status = "" if passed is None else ("PASS" if passed else "FAIL")
        return f"  {label:<34}{fmt(value, unit):>12}{target:>12}{status:>8}"

    def meets(value: float | None, ok) -> bool:
        return value is not None and ok(value)

    line = "=" * 70
    lines = [
        line,
        "  REAL-TIME PERFORMANCE — FaceForensics++ c23 clips at screen size",
        *_run_info(),
        f"  Screen {stats['screen'][0]}x{stats['screen'][1]}, main loop capped at DISPLAY_FPS={config.DISPLAY_FPS}; "
        f"{len(rows)} clips.",
        "  Frame latency = real screen grab + face detection + tracking/queueing per tick.",
        "  Detection latency = face first tracked -> final verdict (incl. queue wait and confirmation).",
        "  Per-clip verdicts and latencies are for the clip's main face (the track seen in most frames).",
        "  App-added = detection latency minus the time inside that face's own BitMind requests.",
        line,
        f"  {'Metric':<34}{'Measured':>12}{'Target':>12}{'Status':>8}",
        "  " + "-" * 66,
        row("Mean FPS", mean_fps, " fps", f">={config.TARGET_FPS}", meets(mean_fps, lambda v: v >= config.TARGET_FPS)),
        row("Lowest FPS (1 s window)", min_fps, " fps", f">={config.TARGET_FPS}",
            meets(min_fps, lambda v: v >= config.TARGET_FPS)),
        row("Mean frame latency", mean_latency, " ms", f"<={config.TARGET_FRAME_LATENCY_MS}",
            meets(mean_latency, lambda v: v <= config.TARGET_FRAME_LATENCY_MS)),
        row("  p95 frame latency", _p95(stages["total"]), " ms", "", None),
        row("  of which screen capture", _mean(stages["capture"]), " ms", "", None),
        row("  of which face detection", _mean(stages["detect"]), " ms", "", None),
        row("  of which tracking + queueing", _mean(stages["track"]), " ms", "", None),
        row("Mean detection latency", mean_final, " ms", f"<={config.TARGET_DETECTION_LATENCY_MS}",
            meets(mean_final, lambda v: v <= config.TARGET_DETECTION_LATENCY_MS)),
        row("  of which inside BitMind requests", _mean(api_time_ms), " ms", "", None),
        row("  p95 detection latency", _p95(final_ms), " ms", "", None),
        row("  max detection latency", max(final_ms, default=None), " ms", "", None),
        row("  mean time to first verdict", _mean(first_ms), " ms", "", None),
        row("App-added detection latency", mean_app, " ms", f"<={config.TARGET_APP_LATENCY_MS}",
            meets(mean_app, lambda v: v <= config.TARGET_APP_LATENCY_MS)),
        row("  p95 app-added latency", _p95(app_ms), " ms", "", None),
        row("Mean CPU (app process, all cores)", mean_cpu, "%", f"<={config.TARGET_CPU_PERCENT}",
            meets(mean_cpu, lambda v: v <= config.TARGET_CPU_PERCENT)),
        row("Peak RAM (app process)", peak_ram, " MB", f"<={config.TARGET_RAM_MB}",
            meets(peak_ram, lambda v: v <= config.TARGET_RAM_MB)),
        "  " + "-" * 66,
        f"  Clips with a verdict: {len(final_ms)}/{len(rows)}, correct: {sum(r['correct'] is True for r in rows)}",
        f"  BitMind calls: {len(stats['api_ms'])} (failed {stats['failures']}, rate-limited {stats['rate_limited']}), "
        f"mean response {fmt(_mean(stats['api_ms']), ' ms')}",
    ]
    if stats["auth_error"]:
        lines.append("  [ERROR] BitMind rejected the API key — check BITMIND_API_KEY / BITMIND_API_KEY_SECONDARY "
                     f"in .env. ({stats['auth_error']})")
    lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser(description="Real-time performance of the detection pipeline on FaceForensics++ clips.")
    parser.add_argument("--dataset", default=config.FF_DATASET_ROOT, help="FaceForensics++_C23 folder.")
    parser.add_argument("--videos-per-subset", type=int, default=config.PERF_VIDEOS_PER_SUBSET,
                        help="Clips played from each subset (first N by name).")
    args = parser.parse_args()
    lock = _hold_test_lock()  # referenced until main() returns, so the lock file stays open and locked

    playlist = _playlist(args.dataset, args.videos_per_subset)
    seconds = 0.0
    for _, _, path in playlist:
        cap = cv2.VideoCapture(path)
        seconds += cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0) + GAP_SECONDS
        cap.release()
    print(f"[PERFORMANCE] {len(playlist)} clips, about {seconds / 60:.1f} min of playback.")

    stats = _run(playlist)
    rows = _clip_rows(playlist, stats["faces"])
    lines = _summary_lines(stats, rows)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(config.RESULTS_DIR, f"performance_videos_{stamp}.csv")
    summary_path = os.path.join(config.RESULTS_DIR, f"performance_summary_{stamp}.txt")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=VIDEO_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"  Per-clip CSV : {csv_path}\n  Summary      : {summary_path}")


if __name__ == "__main__":
    main()
