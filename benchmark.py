"""
benchmark.py

Measures end-to-end real-time detection pipeline performance.
Includes BitMind API response time measurement and handles the case where the API key is not configured.

Usage:
    python benchmark.py
"""

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


def run_benchmark_pass(pass_number: int) -> dict:
    print(f"\n[BENCHMARK] Pass {pass_number}/{config.BENCHMARK_REPEAT}")

    capture = ScreenCapture()
    detector = FaceDetector()
    preprocessor = Preprocessor()

    # Try to load inference engine; skip API calls if key not configured
    try:
        from runtime.detection.inference import DeepfakeInferenceEngine
        engine = DeepfakeInferenceEngine()
        api_enabled = True
        print("  [INFO] BitMind API engine initialised. API calls will be made.")
    except EnvironmentError:
        engine = None
        api_enabled = False
        print("  [WARNING] BITMIND_API_KEY not configured. Benchmarking pipeline without API calls.")

    capture.start()
    process = psutil.Process(os.getpid())

    latencies = []
    fps_samples = []
    cpu_samples = []
    ram_samples = []
    api_ms_samples = []

    total_duration = config.BENCHMARK_WARMUP_SECONDS + config.BENCHMARK_MEASURE_SECONDS
    start_time = time.perf_counter()
    frame_count = 0
    interval_start = time.perf_counter()
    interval_frames = 0

    print(f"  Warming up for {config.BENCHMARK_WARMUP_SECONDS}s...")

    while True:
        elapsed = time.perf_counter() - start_time
        if elapsed >= total_duration:
            break

        in_measurement = elapsed >= config.BENCHMARK_WARMUP_SECONDS
        frame_start = time.perf_counter()

        frame = capture.read()
        face_crops = detector.detect(frame)

        if api_enabled:
            jpeg_list = [preprocessor.process(crop) for crop, _ in face_crops]
            if not jpeg_list:
                dummy_crop = np.zeros((224, 224, 3), dtype=np.uint8)
                jpeg_list = [preprocessor.process(dummy_crop)]
            results = [
                engine.run_single(jpeg, bbox={}, face_id=f"bench_{i}")
                for i, jpeg in enumerate(jpeg_list)
            ]
            if results and in_measurement:
                avg_api_ms = sum(r.api_response_ms for r in results) / len(results)
                api_ms_samples.append(avg_api_ms)

        frame_latency_ms = (time.perf_counter() - frame_start) * 1000

        if in_measurement:
            latencies.append(frame_latency_ms)
            frame_count += 1
            interval_frames += 1

            interval_elapsed = time.perf_counter() - interval_start
            if interval_elapsed >= 1.0:
                fps = interval_frames / interval_elapsed
                cpu = psutil.cpu_percent(interval=None)
                ram = process.memory_info().rss / (1024 * 1024)
                fps_samples.append(fps)
                cpu_samples.append(cpu)
                ram_samples.append(ram)
                interval_frames = 0
                interval_start = time.perf_counter()
        else:
            if elapsed >= config.BENCHMARK_WARMUP_SECONDS - 1:
                print("  Measurement started...")

    detector.close()
    capture.stop()

    results_dict = {
        "mean_fps":        np.mean(fps_samples) if fps_samples else 0.0,
        "min_fps":         np.min(fps_samples) if fps_samples else 0.0,
        "mean_latency_ms": np.mean(latencies) if latencies else 0.0,
        "p95_latency_ms":  np.percentile(latencies, 95) if latencies else 0.0,
        "mean_cpu_pct":    np.mean(cpu_samples) if cpu_samples else 0.0,
        "peak_ram_mb":     np.max(ram_samples) if ram_samples else 0.0,
        "mean_api_ms":     np.mean(api_ms_samples) if api_ms_samples else None,
    }

    print(f"  Pass {pass_number} done — "
          f"FPS={results_dict['mean_fps']:.1f}, "
          f"Latency={results_dict['mean_latency_ms']:.1f}ms, "
          f"CPU={results_dict['mean_cpu_pct']:.1f}%, "
          f"RAM={results_dict['peak_ram_mb']:.0f}MB"
          + (f", API={results_dict['mean_api_ms']:.0f}ms" if results_dict['mean_api_ms'] else ""))
    return results_dict


def print_summary(all_passes: list):
    mean_fps      = np.mean([p["mean_fps"] for p in all_passes])
    mean_latency  = np.mean([p["mean_latency_ms"] for p in all_passes])
    mean_cpu      = np.mean([p["mean_cpu_pct"] for p in all_passes])
    peak_ram      = np.max([p["peak_ram_mb"] for p in all_passes])
    api_values    = [p["mean_api_ms"] for p in all_passes if p["mean_api_ms"] is not None]
    mean_api_ms   = np.mean(api_values) if api_values else None
    fps_variance  = np.std([p["mean_fps"] for p in all_passes]) / mean_fps if mean_fps > 0 else 0

    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<30} {'Measured':>10} {'Target':>12} {'Status':>8}")
    print("-" * 70)

    def row(label, measured, target_str, passed, unit=""):
        status = "PASS" if passed else "FAIL"
        print(f"  {label:<30} {f'{measured:.1f}{unit}':>10} {target_str:>12} {status:>8}")

    row("Mean FPS",         mean_fps,    f">={config.TARGET_FPS}",         mean_fps >= config.TARGET_FPS,         " fps")
    row("Mean Latency",     mean_latency, f"<={config.TARGET_LATENCY_MS}",  mean_latency <= config.TARGET_LATENCY_MS, " ms")
    row("Mean CPU",         mean_cpu,    f"<={config.TARGET_CPU_PERCENT}",  mean_cpu <= config.TARGET_CPU_PERCENT,  "%")
    row("Peak RAM",         peak_ram,    f"<={config.TARGET_RAM_MB}",       peak_ram <= config.TARGET_RAM_MB,       " MB")

    if mean_api_ms is not None:
        print(f"  {'API Response Time':<30} {f'{mean_api_ms:.0f} ms':>10} {'(logged)':>12} {'INFO':>8}")

    print("-" * 70)
    variance_status = "PASS" if fps_variance <= 0.10 else "INVESTIGATE"
    print(f"  {'FPS Variance (<10%)':<30} {f'{fps_variance*100:.1f}%':>10} {'<10%':>12} {variance_status:>8}")
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
