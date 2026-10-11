"""
benchmark.py

Measures end-to-end real-time detection pipeline performance.
Mirrors DetectionWorker's architecture: the main loop (capture → face detection
→ tracking → JPEG preprocessing) runs at DISPLAY_FPS while BitMind API calls
run on a separate thread, so API latency is reported on its own and never
counted against frame FPS/latency. Frame latency is also broken down by
stage (screen capture, face detection, tracking + JPEG) to show where the
time goes. Handles the case where the API key is not configured.

Usage:
    python benchmark.py
"""

import threading
import time
import os
import sys
import numpy as np
import psutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

from runtime.capture.screen_capture import ScreenCapture
from runtime.detection.face_detector import FaceDetector
from runtime.detection.preprocessor import Preprocessor
from runtime.detection.inference import DeepfakeInferenceEngine, BitmindAPIError
from runtime.threads.detection_worker import _FaceTracker

_FRAME_INTERVAL = 1.0 / config.DISPLAY_FPS


def _api_loop(engine, pending: dict, pending_lock: threading.Lock,
              stop: threading.Event, measuring: threading.Event, stats: dict):
    """Round-robin pending faces through the API, as DetectionWorker._api_loop does."""
    while not stop.is_set():
        with pending_lock:
            if pending:
                face_id, (jpeg_bytes, bbox) = next(iter(pending.items()))
                # Move it to the back so other faces get a turn next
                del pending[face_id]
                pending[face_id] = (jpeg_bytes, bbox)
            else:
                face_id = None

        if face_id is None:
            time.sleep(0.05)
            continue

        # Unlike the live app, faces are never locked after their first result,
        # so the API thread stays saturated at the rate limit for the whole run —
        # a worst-case concurrent load on the main loop.
        try:
            result = engine.run_single_attempt(jpeg_bytes, bbox, face_id)
        except BitmindAPIError:
            if measuring.is_set():
                stats["failures"] += 1
            continue
        if measuring.is_set():
            stats["api_ms"].append(result.api_response_ms)


def run_benchmark_pass(pass_number: int) -> dict:
    print(f"\n[BENCHMARK] Pass {pass_number}/{config.BENCHMARK_REPEAT}")

    capture = ScreenCapture()
    detector = FaceDetector()
    preprocessor = Preprocessor()
    tracker = _FaceTracker()

    pending: dict = {}   # face_id -> (jpeg_bytes, bbox)
    pending_lock = threading.Lock()
    stop = threading.Event()
    measuring = threading.Event()
    api_stats = {"api_ms": [], "failures": 0}

    try:
        engine = DeepfakeInferenceEngine()
        print("  [INFO] BitMind API engine initialised. API calls run on a separate thread.")
    except EnvironmentError:
        engine = None
        print("  [WARNING] BITMIND_API_KEY not configured. Benchmarking pipeline without API calls.")

    capture.start()
    frame_h, frame_w = capture.read().shape[:2]
    print(f"  Capture {frame_w}x{frame_h} px; faces detected at <= {config.DETECTION_MAX_WIDTH} px wide.")
    process = psutil.Process(os.getpid())

    api_thread = None
    if engine is not None:
        api_thread = threading.Thread(
            target=_api_loop,
            args=(engine, pending, pending_lock, stop, measuring, api_stats),
            daemon=True,
        )
        api_thread.start()

    latencies = []
    stage_ms = {"capture": [], "detect": []}
    fps_samples = []
    cpu_samples = []
    ram_samples = []

    total_duration = config.BENCHMARK_WARMUP_SECONDS + config.BENCHMARK_MEASURE_SECONDS
    start_time = time.perf_counter()
    interval_start = time.perf_counter()
    interval_frames = 0

    print(f"  Warming up for {config.BENCHMARK_WARMUP_SECONDS}s...")

    try:
        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= total_duration:
                break

            if not measuring.is_set() and elapsed >= config.BENCHMARK_WARMUP_SECONDS:
                print("  Measurement started...")
                # The first cpu_percent(None) call always returns 0.0; prime it so
                # the first real sample covers a full one-second interval.
                psutil.cpu_percent(interval=None)
                interval_start = time.perf_counter()
                interval_frames = 0
                measuring.set()

            frame_start = time.perf_counter()

            frame = capture.read()
            t_captured = time.perf_counter()
            face_crops = detector.detect(frame)[:config.MAX_FACES]
            t_detected = time.perf_counter()
            face_ids = tracker.update([b for _, b in face_crops])
            jpegs = [preprocessor.process(crop) for crop, _ in face_crops]
            alive_ids = tracker.alive_ids()
            with pending_lock:
                for fid, jpeg_bytes, (_, bbox) in zip(face_ids, jpegs, face_crops):
                    pending[fid] = (jpeg_bytes, bbox)
                for fid in [k for k in pending if k not in alive_ids]:
                    del pending[fid]

            frame_latency_ms = (time.perf_counter() - frame_start) * 1000

            if measuring.is_set():
                latencies.append(frame_latency_ms)
                stage_ms["capture"].append((t_captured - frame_start) * 1000)
                stage_ms["detect"].append((t_detected - t_captured) * 1000)
                interval_frames += 1

                interval_elapsed = time.perf_counter() - interval_start
                if interval_elapsed >= 1.0:
                    fps_samples.append(interval_frames / interval_elapsed)
                    cpu_samples.append(psutil.cpu_percent(interval=None))
                    ram_samples.append(process.memory_info().rss / (1024 * 1024))
                    interval_frames = 0
                    interval_start = time.perf_counter()

            # Same tick-rate cap as the live app, so CPU reflects real usage rather than a busy loop
            sleep_time = _FRAME_INTERVAL - (time.perf_counter() - frame_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        stop.set()
        if api_thread is not None:
            api_thread.join(timeout=config.BITMIND_API_TIMEOUT_SEC + 1)
            engine.close()
        detector.close()
        capture.stop()

    api_ms = api_stats["api_ms"]
    results_dict = {
        "mean_fps":        np.mean(fps_samples) if fps_samples else 0.0,
        "min_fps":         np.min(fps_samples) if fps_samples else 0.0,
        "mean_latency_ms": np.mean(latencies) if latencies else 0.0,
        "capture_ms":      np.mean(stage_ms["capture"]) if latencies else 0.0,
        "detect_ms":       np.mean(stage_ms["detect"]) if latencies else 0.0,
        "mean_cpu_pct":    np.mean(cpu_samples) if cpu_samples else 0.0,
        "peak_ram_mb":     np.max(ram_samples) if ram_samples else 0.0,
        "mean_api_ms":     np.mean(api_ms) if api_ms else None,
        "api_calls":       len(api_ms),
        "api_failures":    api_stats["failures"],
        "api_enabled":     engine is not None,
    }

    print(f"  Pass {pass_number} done — "
          f"FPS={results_dict['mean_fps']:.1f}, "
          f"Latency={results_dict['mean_latency_ms']:.1f}ms, "
          f"CPU={results_dict['mean_cpu_pct']:.1f}%, "
          f"RAM={results_dict['peak_ram_mb']:.0f}MB"
          + (f", API={results_dict['mean_api_ms']:.0f}ms "
             f"({results_dict['api_calls']} calls, {results_dict['api_failures']} failed)"
             if results_dict["api_enabled"] and results_dict["mean_api_ms"] is not None else ""))
    return results_dict


def print_summary(all_passes: list):
    mean_fps      = np.mean([p["mean_fps"] for p in all_passes])
    lowest_fps    = np.min([p["min_fps"] for p in all_passes])
    mean_latency  = np.mean([p["mean_latency_ms"] for p in all_passes])
    capture_ms    = np.mean([p["capture_ms"] for p in all_passes])
    detect_ms     = np.mean([p["detect_ms"] for p in all_passes])
    mean_cpu      = np.mean([p["mean_cpu_pct"] for p in all_passes])
    peak_ram      = np.max([p["peak_ram_mb"] for p in all_passes])
    api_values    = [p["mean_api_ms"] for p in all_passes if p["mean_api_ms"] is not None]
    mean_api_ms   = np.mean(api_values) if api_values else None
    api_calls     = sum(p["api_calls"] for p in all_passes)
    api_failures  = sum(p["api_failures"] for p in all_passes)

    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print(f"  Main loop capped at DISPLAY_FPS={config.DISPLAY_FPS}; API on a separate thread.")
    print("=" * 70)
    print(f"  {'Metric':<30} {'Measured':>10} {'Target':>12} {'Status':>8}")
    print("-" * 70)

    def row(label, measured, target_str, passed, unit=""):
        status = "PASS" if passed else "FAIL"
        print(f"  {label:<30} {f'{measured:.1f}{unit}':>10} {target_str:>12} {status:>8}")

    def info(label, value):
        print(f"  {label:<30} {value:>10} {'(logged)':>12} {'INFO':>8}")

    row("Mean FPS",            mean_fps,     f">={config.TARGET_FPS}",         mean_fps >= config.TARGET_FPS,         " fps")
    row("Lowest FPS Observed", lowest_fps,   f">={config.TARGET_FPS}",         lowest_fps >= config.TARGET_FPS,       " fps")
    row("Mean Latency",        mean_latency, f"<={config.TARGET_LATENCY_MS}",  mean_latency <= config.TARGET_LATENCY_MS, " ms")
    info("  of which screen capture",  f"{capture_ms:.1f} ms")
    info("  of which face detection",  f"{detect_ms:.1f} ms")
    info("  of which tracking + JPEG", f"{mean_latency - capture_ms - detect_ms:.1f} ms")
    row("Mean CPU",            mean_cpu,     f"<={config.TARGET_CPU_PERCENT}", mean_cpu <= config.TARGET_CPU_PERCENT, "%")
    row("Peak RAM",            peak_ram,     f"<={config.TARGET_RAM_MB}",      peak_ram <= config.TARGET_RAM_MB,      " MB")

    if mean_api_ms is not None:
        info("API Response Time", f"{mean_api_ms:.0f} ms")
    if all_passes[0]["api_enabled"]:
        info("API Calls (failed)", f"{api_calls} ({api_failures})")
        if api_calls == 0:
            print("  [WARNING] No API calls were made — keep a face visible on screen during the benchmark.")

    print("=" * 70)


def main():
    print("[BENCHMARK] Starting pipeline benchmark.")
    print(f"  Warmup: {config.BENCHMARK_WARMUP_SECONDS}s | "
          f"Measure: {config.BENCHMARK_MEASURE_SECONDS}s | "
          f"Passes: {config.BENCHMARK_REPEAT}")

    all_passes = []
    for i in range(1, config.BENCHMARK_REPEAT + 1):
        result = run_benchmark_pass(i)
        all_passes.append(result)

    print_summary(all_passes)


if __name__ == "__main__":
    main()
