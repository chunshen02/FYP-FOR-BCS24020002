"""Screen capture -> face crop -> face tracking -> BitMind-ready JPEG, on FaceForensics++ (c23) clips shown full screen."""

import os

# Silence MediaPipe / TFLite start-up logs so the progress lines stay readable (same as main.py).
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

import argparse
import csv
import ctypes
import multiprocessing
import statistics
import sys
import time
from collections import Counter
from datetime import datetime

import cv2
import numpy as np

import config
from realtime_performance import GAP_SECONDS, _FRAME_INTERVAL, _VideoScreen, _mean, _p95, _playlist
from runtime.capture.screen_capture import ScreenCapture
from runtime.detection.face_detector import FaceDetector
from runtime.detection.preprocessor import Preprocessor
from runtime.threads.detection_worker import _FaceTracker

WINDOW_NAME = "Screen face pipeline test (Esc to stop)"
QUIT_KEYS = (27, ord("q"))
# Values of the shared "what is on screen" index besides a clip number.
WARMUP, GAP = -2, -1
CLIP_CSV_FIELDS = ["clip", "subset", "video", "frames", "face_hit_rate", "tracks", "track_continuity",
                   "mean_crop_px", "first_payload_ms", "crop_file"]


def _player(playlist: list, width: int, height: int, clip_now, clip_start, ready, start, stop):
    """Child process standing in for the video-call app: shows the playlist full screen on the real display."""
    # The frames are built at mss's physical resolution; without per-monitor DPI awareness Windows
    # display scaling would stretch the window, so the screen would not show them 1:1.
    ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    screen = _VideoScreen(playlist, width, height)
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow(WINDOW_NAME, np.zeros((height, width, 3), dtype=np.uint8))
        cv2.waitKey(1)
        ready.set()
        while not start.is_set():
            if stop.is_set() or cv2.waitKey(50) in QUIT_KEYS:
                return

        shown = None
        while not stop.is_set():
            frame, clip = screen.frame(time.perf_counter())
            if frame is None:
                return
            if frame is not shown:
                cv2.imshow(WINDOW_NAME, frame)
                shown = frame
            key = cv2.waitKey(5)
            # Published only after waitKey has painted the frame, so the pipeline never
            # attributes a grab to a clip that is not on screen yet.
            on_screen = GAP if clip is None else clip
            if on_screen != clip_now.value:
                if clip is not None:
                    clip_start[clip] = time.perf_counter()
                clip_now.value = on_screen
            if key in QUIT_KEYS:
                return
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        screen.close()
        cv2.destroyAllWindows()


def _crop_filename(playlist: list, clip: int | None, face_id: str) -> str:
    if clip is None:
        return f"{face_id}.jpg"
    subset, _, path = playlist[clip]
    return f"clip{clip + 1:02d}_{subset}_{os.path.splitext(os.path.basename(path))[0]}_{face_id}.jpg"


def _clip_progress(index: int, clip: dict) -> str:
    hit_rate = f"{clip['hits'] / clip['frames']:.0%}" if clip["frames"] else "n/a"
    first = f"{clip['first_payload'] * 1000:.0f} ms" if clip["first_payload"] is not None else "n/a"
    return (f"    clip {index + 1}: face in {hit_rate} of frames, {len(clip['ids'])} track(s), "
            f"on screen -> first payload {first}")


def _run(playlist: list, crops_dir: str) -> dict:
    """Show the playlist full screen in a player process and run the app's pipeline on real screen grabs."""
    capture = ScreenCapture()
    capture.start()
    screen_h, screen_w = capture.read().shape[:2]
    clip_now = multiprocessing.Value("i", WARMUP)
    clip_start = multiprocessing.Array("d", len(playlist))
    ready, start, stop = multiprocessing.Event(), multiprocessing.Event(), multiprocessing.Event()
    player = multiprocessing.Process(
        target=_player, args=(playlist, screen_w, screen_h, clip_now, clip_start, ready, start, stop), daemon=True,
    )
    player.start()
    detector = FaceDetector()
    tracker = _FaceTracker()
    preprocessor = Preprocessor()

    clips = [{"frames": 0, "hits": 0, "ids": Counter(), "crop_px": [], "first_payload": None} for _ in playlist]
    stages = {"capture": [], "detect": [], "track": [], "prepare": [], "total": []}
    prep_ms, jpeg_kb, fps_samples = [], [], []
    crop_files = {}   # face_id -> file holding the first JPEG prepared for that track
    gap_frames = gap_hits = window_frames = 0
    window_start = None
    last_on_screen = WARMUP

    try:
        while not ready.wait(0.5):
            if not player.is_alive():
                sys.exit("[ERROR] The video player process exited before opening its window.")
        print(f"  Screen {screen_w}x{screen_h}; first {config.PERF_WARMUP_SECONDS}s are warm-up on a black screen "
              "(not measured).", flush=True)
        loop_start = time.perf_counter()
        while not stop.is_set():
            tick_start = time.perf_counter()
            if not start.is_set() and tick_start - loop_start >= config.PERF_WARMUP_SECONDS:
                start.set()

            before = clip_now.value
            t0 = time.perf_counter()
            frame = capture.read()
            t_captured = time.perf_counter()
            after = clip_now.value
            faces = detector.detect(frame)
            t_detected = time.perf_counter()
            face_ids = tracker.update([b for _, b in faces])
            t_tracked = time.perf_counter()
            # The app prepares only the one crop it submits per second; preparing every
            # face on every frame makes this stage an upper bound.
            payloads, face_prep_ms = [], []
            for crop, _ in faces:
                t_prep = time.perf_counter()
                jpeg_bytes, _ = preprocessor.process_with_array(crop)
                face_prep_ms.append((time.perf_counter() - t_prep) * 1000)
                payloads.append(jpeg_bytes)
            t_end = time.perf_counter()

            # A grab taken while the player switched between a clip and a gap may show
            # either, but a face in it can only have come from the clip.
            owner = after if after >= 0 else (before if before >= 0 else None)
            for fid, jpeg_bytes in zip(face_ids, payloads):
                if fid not in crop_files:
                    crop_files[fid] = _crop_filename(playlist, owner, fid)
                    with open(os.path.join(crops_dir, crop_files[fid]), "wb") as f:
                        f.write(jpeg_bytes)

            if before != WARMUP:
                if window_start is None:
                    window_start = t_end
                stages["capture"].append((t_captured - t0) * 1000)
                stages["detect"].append((t_detected - t_captured) * 1000)
                stages["track"].append((t_tracked - t_detected) * 1000)
                stages["prepare"].append((t_end - t_tracked) * 1000)
                stages["total"].append((t_end - t0) * 1000)
                prep_ms += face_prep_ms
                jpeg_kb += [len(p) / 1024 for p in payloads]
                window_frames += 1
                if t_end - window_start >= 1.0:
                    fps_samples.append(window_frames / (t_end - window_start))
                    window_start, window_frames = t_end, 0
                # Grabs taken while the player switched (before != after) have no single ground truth.
                if before == after == GAP:
                    gap_frames += 1
                    gap_hits += bool(faces)
                elif before == after:
                    clip = clips[after]
                    clip["frames"] += 1
                    if faces:
                        clip["hits"] += 1
                        clip["ids"].update(face_ids)
                        clip["crop_px"].append(min(faces[0][0].shape[:2]))
                        if clip["first_payload"] is None:
                            clip["first_payload"] = t_end - clip_start[after]

            if last_on_screen >= 0 and after != last_on_screen:
                print(_clip_progress(last_on_screen, clips[last_on_screen]), flush=True)
            last_on_screen = after

            # Same tick-rate cap as the live app, so the loop does the same work per second.
            sleep_time = _FRAME_INTERVAL - (time.perf_counter() - tick_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n  Interrupted — summarising what was measured so far.")
    finally:
        stop.set()
        player.join(timeout=5)
        detector.close()
        capture.stop()

    return {
        "screen": (screen_w, screen_h), "stages": stages, "fps": fps_samples, "clips": clips,
        "gap_frames": gap_frames, "gap_hits": gap_hits, "prep_ms": prep_ms, "jpeg_kb": jpeg_kb,
        "crop_files": crop_files,
    }


def _clip_rows(playlist: list, stats: dict) -> list:
    """One row per clip that was shown; crop_file is the first JPEG of its main face (the track seen in most frames)."""
    rows = []
    for i, ((subset, _, path), clip) in enumerate(zip(playlist, stats["clips"])):
        if not clip["frames"]:
            continue
        main_id, main_frames = clip["ids"].most_common(1)[0] if clip["ids"] else (None, 0)
        rows.append({
            "clip": i + 1, "subset": subset, "video": os.path.basename(path),
            "frames": clip["frames"], "face_hit_rate": round(clip["hits"] / clip["frames"], 3),
            "tracks": len(clip["ids"]),
            "track_continuity": round(main_frames / clip["hits"], 3) if clip["hits"] else "",
            "mean_crop_px": round(statistics.mean(clip["crop_px"])) if clip["crop_px"] else "",
            "first_payload_ms": round(clip["first_payload"] * 1000) if clip["first_payload"] is not None else "",
            "crop_file": stats["crop_files"].get(main_id, ""),
        })
    return rows


def _percent(numerator: int, denominator: int) -> float | None:
    return 100 * numerator / denominator if denominator else None


def _summary_lines(stats: dict, rows: list) -> list:
    stages = stats["stages"]
    clips = [c for c in stats["clips"] if c["frames"]]
    hits = sum(c["hits"] for c in clips)
    hit_rate = _percent(hits, sum(c["frames"] for c in clips))
    continuity = _percent(sum(c["ids"].most_common(1)[0][1] for c in clips if c["ids"]), hits)
    gap_rate = _percent(stats["gap_hits"], stats["gap_frames"])
    first_payload = [r["first_payload_ms"] for r in rows if r["first_payload_ms"] != ""]
    mean_fps, min_fps = _mean(stats["fps"]), min(stats["fps"], default=None)
    mean_latency = _mean(stages["total"])

    def fmt(value: float | None, unit: str) -> str:
        return "n/a" if value is None else f"{value:.1f}{unit}"

    def row(label: str, value: float | None, unit: str, target: str = "", passed: bool | None = None) -> str:
        status = "" if passed is None else ("PASS" if passed else "FAIL")
        return f"  {label:<44}{fmt(value, unit):>12}{target:>10}{status:>8}"

    def meets(value: float | None, ok) -> bool:
        return value is not None and ok(value)

    line = "=" * 76
    return [
        line,
        "  SCREEN FACE PIPELINE — FaceForensics++ c23 clips shown full screen",
        f"  Screen {stats['screen'][0]}x{stats['screen'][1]}. A player process showed {len(rows)} clips full screen; the pipeline",
        f"  grabbed the real screen at DISPLAY_FPS={config.DISPLAY_FPS}, detected, cropped and tracked",
        "  faces on the grab, and turned every face into the JPEG the app submits to",
        "  BitMind. BitMind is not called: API speed is measured by",
        "  realtime_performance.py and accuracy by detection_accuracy.py.",
        "  'On screen' times start when a clip's first frame was drawn (+/- one refresh).",
        line,
        f"  {'Metric':<44}{'Measured':>12}{'Target':>10}{'Status':>8}",
        "  " + "-" * 74,
        "  1. Screen capture (mss)",
        row("Mean FPS", mean_fps, " fps", f">={config.TARGET_FPS}", meets(mean_fps, lambda v: v >= config.TARGET_FPS)),
        row("Lowest FPS (1 s window)", min_fps, " fps", f">={config.TARGET_FPS}",
            meets(min_fps, lambda v: v >= config.TARGET_FPS)),
        row("Mean screen grab", _mean(stages["capture"]), " ms"),
        row("  p95 screen grab", _p95(stages["capture"]), " ms"),
        "  2. Face detection + cropping (MediaPipe + OpenCV, on the grab)",
        row("Face hit rate (clip frames with a face)", hit_rate, "%", f">={config.TARGET_FACE_HIT_RATE:.0%}",
            meets(hit_rate, lambda v: v >= 100 * config.TARGET_FACE_HIT_RATE)),
        row("False detections (black gap frames)", gap_rate, "%", "0%", meets(gap_rate, lambda v: v == 0)),
        row("Mean detect + crop", _mean(stages["detect"]), " ms"),
        row("  p95 detect + crop", _p95(stages["detect"]), " ms"),
        row("Mean face crop size (shorter side)", _mean([px for c in clips for px in c["crop_px"]]), " px"),
        "  3. Face tracking",
        row("Main-face track continuity", continuity, "%", f">={config.TARGET_TRACK_CONTINUITY:.0%}",
            meets(continuity, lambda v: v >= 100 * config.TARGET_TRACK_CONTINUITY)),
        row("Mean tracks per clip", _mean([r["tracks"] for r in rows]), ""),
        row("Mean tracking", _mean(stages["track"]), " ms"),
        "  4. BitMind payload preparation (not sent)",
        row("Mean per face (resize + JPEG)", _mean(stats["prep_ms"]), " ms"),
        row("  p95 per face", _p95(stats["prep_ms"]), " ms"),
        row("Mean JPEG size", _mean(stats["jpeg_kb"]), " KB"),
        "  5. Real-time latency",
        row("Mean frame latency (grab -> payloads ready)", mean_latency, " ms", f"<={config.TARGET_FRAME_LATENCY_MS}",
            meets(mean_latency, lambda v: v <= config.TARGET_FRAME_LATENCY_MS)),
        row("  p95 frame latency", _p95(stages["total"]), " ms"),
        row("  max frame latency", max(stages["total"], default=None), " ms"),
        row("Mean on screen -> first payload ready", _mean(first_payload), " ms"),
        row("  p95 on screen -> first payload ready", _p95(first_payload), " ms"),
        "  " + "-" * 74,
        f"  Frames measured: {len(stages['total'])}; payloads prepared: {len(stats['prep_ms'])} (every face on",
        "  every frame — the app prepares only the one it submits per second, so",
        "  frame latency here is an upper bound).",
        line,
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Screen capture -> face crop -> tracking -> BitMind-ready JPEG test on FaceForensics++ clips "
                    "shown full screen (BitMind is not called).")
    parser.add_argument("--dataset", default=config.FF_DATASET_ROOT, help="FaceForensics++_C23 folder.")
    parser.add_argument("--videos-per-subset", type=int, default=config.PIPELINE_VIDEOS_PER_SUBSET,
                        help="Clips shown from each subset (first N by name).")
    args = parser.parse_args()

    playlist = _playlist(args.dataset, args.videos_per_subset)
    seconds = config.PERF_WARMUP_SECONDS
    for _, _, path in playlist:
        cap = cv2.VideoCapture(path)
        seconds += cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25.0) + GAP_SECONDS
        cap.release()
    print(f"[PIPELINE] {len(playlist)} clips, about {seconds / 60:.1f} min. The clips play full screen on the "
          "primary monitor; press Esc in the video window to stop early.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    crops_dir = os.path.join(config.RESULTS_DIR, f"pipeline_crops_{stamp}")
    os.makedirs(crops_dir, exist_ok=True)
    stats = _run(playlist, crops_dir)
    rows = _clip_rows(playlist, stats)
    lines = _summary_lines(stats, rows)

    csv_path = os.path.join(config.RESULTS_DIR, f"pipeline_clips_{stamp}.csv")
    summary_path = os.path.join(config.RESULTS_DIR, f"pipeline_summary_{stamp}.txt")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CLIP_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"  Per-clip CSV : {csv_path}\n  Summary      : {summary_path}\n  Face crops   : {crops_dir}")


if __name__ == "__main__":
    main()
