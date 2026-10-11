"""
tracking_eval.py

Offline evaluation of face-tracking ID stability (_FaceTracker) on a
FaceForensics++ dataset. FF++ has no per-frame identity annotations, so
stability is measured by proxy: most clips show one main subject, whose
face should keep a single ID for the whole clip.

Each video goes through the same detection front end as DetectionWorker
(DETECTION_MAX_WIDTH, MAX_FACES largest faces) and is sampled at roughly
DISPLAY_FPS, so the tracker sees the same per-tick motion as in the live
app. No BitMind API calls are made.

Per-video metrics:
  - main_id_share : share of face-bearing ticks in which the most frequent ID
                    is present (near 100% when the subject keeps one ID).
  - breaks        : a newly created ID whose box overlaps the last box of a
                    track that is absent this tick — the same face re-IDed.
      tracker break : the old track was still within FACE_TRACK_GRACE_FRAMES,
                      so it should have matched (a tracker failure).
      gap break     : the face was undetected for longer than the grace
                      period, so a new ID is by design (a detector miss).
  - max_faces / multi_face_rate : extra detections, typically detector false
                    positives in these single-subject clips.

Usage:
    python tracking_eval.py <ff_root> [--subsets original DeepFakeDetection Deepfakes]
        [--workers N] [--limit N] [--csv-path path]
    python tracking_eval.py --summary-only [--csv-path path]

Per-video results are appended to the CSV as each video finishes, so an
interrupted run resumes where it stopped when re-run with the same CSV.
"""

import argparse
import csv
import multiprocessing
import os
import statistics
import sys
import time
from collections import Counter

import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from runtime.detection.face_detector import FaceDetector
from runtime.threads.detection_worker import _FaceTracker, _iou

DEFAULT_SUBSETS = ["original", "DeepFakeDetection", "Deepfakes"]
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracking_eval_results.csv")
CSV_FIELDS = [
    "subset", "video", "frames", "step", "face_rate", "main_id_share", "tracker_breaks", "gap_breaks",
    "ids_total", "max_faces", "multi_face_rate", "seconds", "error",
]
INT_FIELDS = ("frames", "step", "tracker_breaks", "gap_breaks", "ids_total", "max_faces")
FLOAT_FIELDS = ("face_rate", "main_id_share", "multi_face_rate", "seconds")
STABLE_SHARE = 0.95
WORST_VIDEOS_SHOWN = 10

_detector = None  # one FaceDetector per worker process


def _init_worker():
    global _detector
    _detector = FaceDetector()


def _evaluate_video(task: tuple) -> dict:
    """Track faces through one video and return its per-video stability metrics."""
    subset, path = task
    row = {"subset": subset, "video": os.path.basename(path), "error": ""}
    t_start = time.perf_counter()

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        row["error"] = "cannot open"
        return row
    fps = cap.get(cv2.CAP_PROP_FPS) or config.DISPLAY_FPS
    step = max(1, round(fps / config.DISPLAY_FPS))

    tracker = _FaceTracker()
    last_seen: dict = {}       # face_id -> (tick, bbox) of its latest appearance
    presence = Counter()       # face_id -> ticks present
    tick = face_ticks = multi_ticks = max_faces = 0
    tracker_breaks = gap_breaks = 0

    frame_idx = 0
    while cap.grab():
        if frame_idx % step == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            bboxes = [b for _, b in _detector.detect(frame)[:config.MAX_FACES]]
            ids = tracker.update(bboxes)
            if ids:
                face_ticks += 1
                multi_ticks += len(ids) > 1
                max_faces = max(max_faces, len(ids))
            for fid, bbox in zip(ids, bboxes):
                if fid not in last_seen:
                    lost = [t for t, b in last_seen.values() if t < tick and _iou(bbox, b) > 0]
                    if lost:
                        if tick - max(lost) - 1 > config.FACE_TRACK_GRACE_FRAMES:
                            gap_breaks += 1
                        else:
                            tracker_breaks += 1
                last_seen[fid] = (tick, bbox)
                presence[fid] += 1
            tick += 1
        frame_idx += 1
    cap.release()

    row.update({
        "frames": tick,
        "step": step,
        "face_rate": round(face_ticks / tick, 4) if tick else 0.0,
        "main_id_share": round(presence.most_common(1)[0][1] / face_ticks, 4) if face_ticks else 0.0,
        "tracker_breaks": tracker_breaks,
        "gap_breaks": gap_breaks,
        "ids_total": len(presence),
        "max_faces": max_faces,
        "multi_face_rate": round(multi_ticks / tick, 4) if tick else 0.0,
        "seconds": round(time.perf_counter() - t_start, 1),
    })
    if tick == 0:
        row["error"] = "no frames decoded"
    return row


def _load_rows(csv_path: str) -> list:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _list_tasks(ff_root: str, subsets: list, limit: int | None) -> list:
    tasks = []
    for subset in subsets:
        folder = os.path.join(ff_root, subset)
        if not os.path.isdir(folder):
            print(f"[ERROR] Subset folder not found: {folder}")
            sys.exit(1)
        videos = sorted(f for f in os.listdir(folder) if f.lower().endswith(".mp4"))[:limit]
        tasks += [(subset, os.path.join(folder, v)) for v in videos]
    return tasks


def _run(tasks: list, csv_path: str, workers: int):
    write_header = not os.path.exists(csv_path)
    t_start = time.perf_counter()
    with open(csv_path, "a", newline="", encoding="utf-8") as f, \
            multiprocessing.Pool(workers, initializer=_init_worker) as pool:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for done, row in enumerate(pool.imap_unordered(_evaluate_video, tasks), start=1):
            writer.writerow(row)
            f.flush()
            elapsed = time.perf_counter() - t_start
            eta_min = elapsed / done * (len(tasks) - done) / 60
            status = row["error"] or (f"main-ID {row['main_id_share'] * 100:.0f}%, "
                                      f"breaks {row['tracker_breaks']}+{row['gap_breaks']}gap")
            print(f"  [{done}/{len(tasks)}] {row['subset']}/{row['video']}: {status}  "
                  f"(elapsed {elapsed / 60:.1f} min, ETA {eta_min:.0f} min)", flush=True)


def _pct(part: int, whole: int) -> str:
    return f"{part / whole * 100:.1f}%" if whole else "n/a"


def _count_line(rows: list, predicate) -> str:
    k = sum(1 for r in rows if predicate(r))
    return f"{k} ({_pct(k, len(rows))})"


def _subset_stats(rows: list) -> list:
    """Return (label, value) lines describing one group of per-video rows."""
    n = len(rows)
    total_ticks = sum(r["frames"] for r in rows)
    total_tracker = sum(r["tracker_breaks"] for r in rows)
    return [
        ("Videos", f"{n}"),
        ("Mean face detection rate", f"{statistics.mean(r['face_rate'] for r in rows) * 100:.1f}%"),
        ("Mean main-ID share", f"{statistics.mean(r['main_id_share'] for r in rows) * 100:.1f}%"),
        (f"Videos with main-ID share >= {STABLE_SHARE:.0%}", _count_line(rows, lambda r: r["main_id_share"] >= STABLE_SHARE)),
        ("Videos with 0 tracker breaks", _count_line(rows, lambda r: r["tracker_breaks"] == 0)),
        ("Tracker breaks / video", f"mean {total_tracker / n:.2f}, median "
                                   f"{statistics.median(r['tracker_breaks'] for r in rows):.0f}, "
                                   f"max {max(r['tracker_breaks'] for r in rows)}"),
        ("Tracker breaks / 1000 ticks", f"{total_tracker / total_ticks * 1000:.2f}" if total_ticks else "n/a"),
        ("Gap breaks / video", f"mean {statistics.mean(r['gap_breaks'] for r in rows):.2f}, "
                               f"max {max(r['gap_breaks'] for r in rows)}"),
        ("Videos with extra detections", _count_line(rows, lambda r: r["max_faces"] > 1)),
        ("Mean ticks with >1 detection", f"{statistics.mean(r['multi_face_rate'] for r in rows) * 100:.1f}%"),
        ("Mean IDs assigned / video", f"{statistics.mean(r['ids_total'] for r in rows):.2f}"),
    ]


def _print_block(title: str, rows: list):
    print(f"\n  {title}")
    print(f"  {'-' * 66}")
    if not rows:
        print("    (no videos)")
        return
    for label, value in _subset_stats(rows):
        print(f"    {label:<34} {value}")


def _print_paired_comparison(by_subset: dict, fake_subset: str):
    """Compare each manipulated video with the original clip it was generated from (FF++ name '<target>_<source>.mp4')."""
    originals = {os.path.splitext(r["video"])[0]: r for r in by_subset.get("original", [])}
    pairs = [
        (originals[r["video"].split("_")[0]], r)
        for r in by_subset.get(fake_subset, [])
        if r["video"].split("_")[0] in originals
    ]
    print(f"\n  PAIRED COMPARISON: original vs {fake_subset} (same source clip)")
    print(f"  {'-' * 66}")
    if not pairs:
        print("    (no matching pairs)")
        return
    n = len(pairs)
    more = sum(d["tracker_breaks"] > o["tracker_breaks"] for o, d in pairs)
    fewer = sum(d["tracker_breaks"] < o["tracker_breaks"] for o, d in pairs)
    print(f"    {'Pairs':<34} {n}")
    print(f"    {'Identical tracker breaks':<34} {_count_line(pairs, lambda p: p[0]['tracker_breaks'] == p[1]['tracker_breaks'])}")
    print(f"    {'Fake more / fewer tracker breaks':<34} {more} / {fewer}")
    print(f"    {'Mean tracker-break diff (fake-orig)':<34} "
          f"{statistics.mean(d['tracker_breaks'] - o['tracker_breaks'] for o, d in pairs):+.2f}")
    print(f"    {'Mean main-ID share diff':<34} "
          f"{statistics.mean(d['main_id_share'] - o['main_id_share'] for o, d in pairs) * 100:+.2f} pp")
    print(f"    {'Mean face detection rate diff':<34} "
          f"{statistics.mean(d['face_rate'] - o['face_rate'] for o, d in pairs) * 100:+.2f} pp")


def print_summary(raw_rows: list):
    """Print per-subset stability statistics, a paired original-vs-fake comparison, and the worst videos."""
    errors = [r for r in raw_rows if r["error"]]
    rows = [
        {**r, **{k: int(r[k]) for k in INT_FIELDS}, **{k: float(r[k]) for k in FLOAT_FIELDS}}
        for r in raw_rows if not r["error"]
    ]
    by_subset: dict = {}
    for r in rows:
        by_subset.setdefault(r["subset"], []).append(r)

    print("\n" + "=" * 70)
    print("  FACE TRACKING STABILITY SUMMARY")
    print(f"  Sampled at ~DISPLAY_FPS={config.DISPLAY_FPS}, DETECTION_MAX_WIDTH={config.DETECTION_MAX_WIDTH}, "
          f"MAX_FACES={config.MAX_FACES}, FACE_CONFIDENCE_THRESHOLD={config.FACE_CONFIDENCE_THRESHOLD}")
    print(f"  Tracker: IoU>={config.FACE_TRACK_IOU_THRESHOLD}, max dist {config.FACE_TRACK_MAX_DIST_PX}px, "
          f"grace {config.FACE_TRACK_GRACE_FRAMES} ticks")
    print(f"  Videos evaluated: {len(rows)}  (errors: {len(errors)}), "
          f"processing time {sum(r['seconds'] for r in rows) / 3600:.1f} CPU-worker hours")
    print("=" * 70)

    for subset, subset_rows in by_subset.items():
        print(f"\n  [{subset}]")
        _print_block("All videos", subset_rows)
        # Clips with no extra detections isolate the tracker from detector false positives.
        _print_block("Clean videos only (never >1 detection in a tick)",
                     [r for r in subset_rows if r["max_faces"] <= 1])

    for fake_subset in by_subset:
        if fake_subset not in ("original", "DeepFakeDetection"):
            _print_paired_comparison(by_subset, fake_subset)

    worst = sorted(rows, key=lambda r: r["tracker_breaks"], reverse=True)[:WORST_VIDEOS_SHOWN]
    print(f"\n  MOST TRACKER BREAKS (top {WORST_VIDEOS_SHOWN}, for manual review)")
    print(f"  {'-' * 66}")
    for r in worst:
        print(f"    {r['subset'] + '/' + r['video']:<52} {r['tracker_breaks']:>3} breaks, "
              f"main-ID {r['main_id_share'] * 100:.0f}%, max {r['max_faces']} faces")
    for r in errors:
        print(f"    [ERROR] {r['subset']}/{r['video']}: {r['error']}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Offline face-tracking ID-stability evaluation on FaceForensics++.")
    parser.add_argument("ff_root", nargs="?", help="FaceForensics++ root; not needed with --summary-only.")
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Max videos per subset (for quick trial runs).")
    parser.add_argument("--csv-path", default=DEFAULT_CSV_PATH)
    parser.add_argument("--summary-only", action="store_true", help="Skip evaluation; summarise the existing CSV.")
    args = parser.parse_args()

    if not args.summary_only:
        if args.ff_root is None:
            parser.error("ff_root is required unless --summary-only is given")
        done = {(r["subset"], r["video"]) for r in _load_rows(args.csv_path)}
        tasks = [t for t in _list_tasks(args.ff_root, args.subsets, args.limit)
                 if (t[0], os.path.basename(t[1])) not in done]
        print(f"[TRACKING EVAL] {len(tasks)} videos to process ({len(done)} already in {args.csv_path}), "
              f"{args.workers} workers.")
        if tasks:
            _run(tasks, args.csv_path, args.workers)

    rows = _load_rows(args.csv_path)
    if not rows:
        print("[ERROR] No results to summarise.")
        sys.exit(1)
    print_summary(rows)


if __name__ == "__main__":
    main()
