"""
benchmark.py

Measures end-to-end real-time detection pipeline performance.
Compares results against the targets defined in config.py.

Usage:
    python benchmark.py
"""

import time
import os
import sys
import numpy as np
import psutil
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

from runtime.capture.screen_capture import ScreenCapture
from runtime.detection.face_detector import FaceDetector
from runtime.detection.preprocessor import Preprocessor
from runtime.detection.inference import DeepfakeInferenceEngine


def run_benchmark_pass(pass_number: int) -> dict:
    """Run a single benchmark pass and return metric results."""
    print(f"\n[BENCHMARK] Pass {pass_number}/{config.BENCHMARK_REPEAT}")

    capture = ScreenCapture()
    detector = FaceDetector()
    preprocessor = Preprocessor()

    if not os.path.exists(config.ONNX_MODEL_PATH):
        print("[WARNING] ONNX model not found. Using dummy inference for benchmarking.")
        engine = None
    else:
        engine = DeepfakeInferenceEngine()

    capture.start()

    process = psutil.Process(os.getpid())
    latencies = []
    fps_samples = []
    cpu_samples = []
    ram_samples = []

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

        # Pipeline
        frame = capture.read()
        detections = detector.detect(frame)
        face_crops = [d[0] for d in detections]
        tensors = [preprocessor.process(crop) for crop in face_crops]
        if engine and tensors:
            engine.run(tensors)
        elif not detections:
            # No face detected — run inference on a blank tensor for latency measurement
            dummy = preprocessor.process(np.zeros((224, 224, 3), dtype=np.uint8))
            if engine:
                engine.run([dummy])

        frame_end = time.perf_counter()
        frame_latency_ms = (frame_end - frame_start) * 1000

        if in_measurement:
            latencies.append(frame_latency_ms)
            frame_count += 1
            interval_frames += 1

            # Log metrics every second
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
                print(f"  Measurement started...")

    if engine:
        engine.reset_ema()
    detector.close()
    capture.stop()

    results = {
        "mean_fps":        np.mean(fps_samples) if fps_samples else 0.0,
        "min_fps":         np.min(fps_samples) if fps_samples else 0.0,
        "mean_latency_ms": np.mean(latencies) if latencies else 0.0,
        "p95_latency_ms":  np.percentile(latencies, 95) if latencies else 0.0,
        "mean_cpu_pct":    np.mean(cpu_samples) if cpu_samples else 0.0,
        "peak_ram_mb":     np.max(ram_samples) if ram_samples else 0.0,
    }

    print(f"  Pass {pass_number} complete: "
          f"FPS={results['mean_fps']:.1f}, "
          f"Latency={results['mean_latency_ms']:.1f}ms, "
          f"CPU={results['mean_cpu_pct']:.1f}%, "
          f"RAM={results['peak_ram_mb']:.0f}MB")

    return results


def print_summary(all_passes: list):
    """Print a final summary table with PASS/FAIL against targets."""
    mean_fps = np.mean([p["mean_fps"] for p in all_passes])
    mean_latency = np.mean([p["mean_latency_ms"] for p in all_passes])
    mean_cpu = np.mean([p["mean_cpu_pct"] for p in all_passes])
    peak_ram = np.max([p["peak_ram_mb"] for p in all_passes])

    fps_variance = np.std([p["mean_fps"] for p in all_passes]) / mean_fps if mean_fps > 0 else 0

    print("\n" + "=" * 65)
    print("  BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"  {'Metric':<28} {'Measured':>10} {'Target':>10} {'Status':>8}")
    print("-" * 65)

    def row(label, measured, target, pass_condition, unit=""):
        status = "PASS" if pass_condition else "FAIL"
        print(f"  {label:<28} {f'{measured:.1f}{unit}':>10} {f'{target}{unit}':>10} {status:>8}")

    row("Mean FPS",         mean_fps,     f">={config.TARGET_FPS}",          mean_fps >= config.TARGET_FPS,         " fps")
    row("Mean Latency",     mean_latency, f"<={config.TARGET_LATENCY_MS}",   mean_latency <= config.TARGET_LATENCY_MS, " ms")
    row("Mean CPU",         mean_cpu,     f"<={config.TARGET_CPU_PERCENT}",  mean_cpu <= config.TARGET_CPU_PERCENT,  "%")
    row("Peak RAM",         peak_ram,     f"<={config.TARGET_RAM_MB}",       peak_ram <= config.TARGET_RAM_MB,       " MB")

    print("-" * 65)
    variance_status = "PASS" if fps_variance <= 0.10 else "INVESTIGATE"
    print(f"  {'FPS Variance (<10%)':<28} {f'{fps_variance*100:.1f}%':>10} {'<10%':>10} {variance_status:>8}")
    print("=" * 65)


def main():
    print("[BENCHMARK] Starting end-to-end pipeline benchmark.")
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
