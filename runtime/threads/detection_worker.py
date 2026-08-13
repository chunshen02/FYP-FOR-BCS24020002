"""
detection_worker.py

QThread that runs the real-time detection pipeline continuously.
Pipeline: Screen Capture → Face Detection → JPEG Preprocessing → BitMind API
Emits raw frames + FaceResult lists + pending bboxes to the GUI thread.

Architecture:
  Main loop   — capture + face detection at the configured display rate;
                uses cached API results.
  API thread  — processes one face per second (rate limit), round-robin across
                all currently visible faces so every participant gets updated.
  _FaceTracker — assigns stable IDs to faces across frames so result caching
                remains consistent even as faces reorder.
"""

import threading
import time
from dataclasses import replace
import numpy as np
import cv2
from PyQt6.QtCore import QThread, pyqtSignal
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from runtime.capture.screen_capture import ScreenCapture
from runtime.capture import window_affinity
from runtime.detection.face_detector import FaceDetector
from runtime.detection.preprocessor import Preprocessor
from runtime.detection.inference import DeepfakeInferenceEngine, BitmindAPIError, BitmindAuthError
from runtime.detection.explainer import analyze_artifacts
from runtime.storage.database import log_event as db_log_event
import config

_FRAME_INTERVAL = 1.0 / config.DISPLAY_FPS


def _centroid(box: dict) -> tuple:
    return (box["x"] + box["w"] / 2, box["y"] + box["h"] / 2)


def _iou(a: dict, b: dict) -> float:
    """Intersection-over-union of two {'x','y','w','h'} boxes."""
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    union = a["w"] * a["h"] + b["w"] * b["h"] - intersection
    return intersection / union if union > 0 else 0.0


def match_tracks(tracks: dict, bboxes: list, max_dist: float,
                  iou_threshold: float, grace_frames: int,
                  protected_ids: frozenset = frozenset()) -> tuple:
    """
    Match bboxes to existing tracks: highest-IoU-above-threshold first
    (robust to fast movement, since box overlap survives large centroid
    jumps), nearest-centroid-within-max_dist fallback for boxes with no
    overlap. Pure function — no Qt/PyQt dependency — so it is independently
    testable outside the QThread pipeline.

    Args:
        tracks: face_id -> {"bbox": {"x","y","w","h"}, "missed": int}
        bboxes: list of {"x","y","w","h"} dicts for faces in the current frame
        protected_ids: track ids that must survive this update regardless of
            missed count — used to keep a track alive while its BitMind
            request is still in flight, so a late response has a face to
            attach to instead of being orphaned.

    Returns:
        (matched_ids, survivors)
        matched_ids: list[str | None], parallel to bboxes — None means the
            caller must assign a new id for that bbox.
        survivors: tracks to carry into the next frame — matched tracks
            reset to missed=0 with the new bbox; unmatched tracks age by
            one frame and are evicted once missed exceeds grace_frames,
            keeping a face's identity (and cached result) alive across a
            brief occlusion instead of forcing a fresh re-classification.
    """
    used_ids: set = set()
    matched_ids: list = [None] * len(bboxes)

    # Pass 1: IoU, strongest overlap first across all (bbox, track) pairs,
    # so no bbox claims a weak match before a stronger one elsewhere is seen.
    iou_candidates = sorted(
        (
            (_iou(bbox, track["bbox"]), i, tid)
            for i, bbox in enumerate(bboxes)
            for tid, track in tracks.items()
            if _iou(bbox, track["bbox"]) >= iou_threshold
        ),
        key=lambda c: c[0], reverse=True,
    )
    for _, i, tid in iou_candidates:
        if matched_ids[i] is not None or tid in used_ids:
            continue
        matched_ids[i] = tid
        used_ids.add(tid)

    # Pass 2: nearest-centroid fallback for bboxes with no IoU overlap
    for i, bbox in enumerate(bboxes):
        if matched_ids[i] is not None:
            continue
        cx, cy = _centroid(bbox)
        best_id, best_dist = None, max_dist
        for tid, track in tracks.items():
            if tid in used_ids:
                continue
            tcx, tcy = _centroid(track["bbox"])
            dist = ((cx - tcx) ** 2 + (cy - tcy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best_id = dist, tid
        if best_id is not None:
            matched_ids[i] = best_id
            used_ids.add(best_id)

    survivors = {}
    for i, tid in enumerate(matched_ids):
        if tid is not None:
            survivors[tid] = {"bbox": bboxes[i], "missed": 0}
    for tid, track in tracks.items():
        if tid in survivors:
            continue
        missed = track["missed"] + 1
        if missed <= grace_frames or tid in protected_ids:
            survivors[tid] = {"bbox": track["bbox"], "missed": missed}

    return matched_ids, survivors


class _FaceTracker:
    """Assigns stable IDs to faces across frames, tolerating brief occlusion via a grace period."""

    def __init__(self):
        self._tracks: dict = {}   # face_id -> {"bbox": dict, "missed": int}
        self._counter: int = 0

    def update(self, bboxes: list, protected_ids: frozenset = frozenset()) -> list:
        """Match bboxes to existing tracks; assign new IDs to unmatched faces."""
        matched_ids, survivors = match_tracks(
            self._tracks, bboxes,
            config.FACE_TRACK_MAX_DIST_PX, config.FACE_TRACK_IOU_THRESHOLD, config.FACE_TRACK_GRACE_FRAMES,
            protected_ids,
        )

        ids = []
        for i, tid in enumerate(matched_ids):
            if tid is None:
                tid = f"f{self._counter}"
                self._counter += 1
                survivors[tid] = {"bbox": bboxes[i], "missed": 0}
            ids.append(tid)

        self._tracks = survivors
        return ids

    def alive_ids(self) -> set:
        """IDs currently tracked, including those still within their grace period after going undetected."""
        return set(self._tracks.keys())


class DetectionWorker(QThread):
    """
    Runs the full deepfake detection pipeline in a background thread.

    Signals:
        frame_ready(np.ndarray, list, list, list): BGR frame, list of FaceResult
            (locked — classified once, awaiting manual re-check to run again),
            list of pending bboxes (never yet classified), list of rechecking
            bboxes (previously classified, a fresh classification is in flight).
        fps_update(float): Current processing FPS.
        api_response_update(float): Most recent BitMind API response time (ms).
        error_occurred(str): Error message if the pipeline crashes.
    """

    frame_ready = pyqtSignal(object, object, object, object)
    fps_update = pyqtSignal(float)
    api_response_update = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    low_light_warning = pyqtSignal(bool)
    face_count_update = pyqtSignal(int, int)   # (shown, total_detected)
    degraded_mode = pyqtSignal(bool)           # True when the BitMind API is unreachable

    def __init__(self, region: dict = None, face_detector=None, engine=None, parent=None):
        super().__init__(parent)
        self._region = region
        self._running = False
        # Shared state between main loop and API thread
        self._pending_lock = threading.Lock()
        self._pending_faces: dict = {}   # face_id -> (jpeg_bytes, bbox, resized_bgr)
        self._result_lock = threading.Lock()
        self._cached_results: dict = {}  # face_id -> FaceResult
        # A face with a result is "locked" — excluded from auto re-submission
        # until request_recheck() clears it. _recheck_ids tracks faces whose
        # lock was just cleared, purely so the GUI can show "Re-checking..."
        # instead of "Analyzing..." for a face that already has a result.
        self._locked_ids: set = set()
        self._recheck_ids: set = set()
        self._inflight_lock = threading.Lock()
        self._inflight_id: str | None = None   # face currently mid-HTTP-call, protected from track eviction
        self._excl_lock = threading.Lock()
        self._excl_rect: tuple | None = None  # (x, y, w, h) physical pixels
        self._focus_lock = threading.Lock()
        self._focus_rect: tuple | None = None  # (x, y, w, h) frame pixels
        # Borrowed references — caller owns lifetime; None means create locally
        self._shared_detector = face_detector
        self._shared_engine = engine

    def run(self):
        self._running = True
        self._pending_faces.clear()
        self._cached_results.clear()
        self._locked_ids.clear()
        self._recheck_ids.clear()
        self._inflight_id = None

        capture = ScreenCapture(region=self._region)
        preprocessor = Preprocessor()
        tracker = _FaceTracker()

        owned_detector = self._shared_detector is None
        detector = self._shared_detector or FaceDetector()

        owned_engine = self._shared_engine is None
        if owned_engine:
            try:
                engine = DeepfakeInferenceEngine()
            except EnvironmentError as e:
                self.error_occurred.emit(str(e))
                if owned_detector:
                    detector.close()
                return
        else:
            engine = self._shared_engine

        capture.start()

        api_thread = threading.Thread(
            target=self._api_loop,
            args=(engine,),
            daemon=True,
        )
        api_thread.start()

        frame_count = 0
        fps_timer_start = time.perf_counter()

        try:
            while self._running:
                t_frame_start = time.perf_counter()

                # Shared with the manual-screenshot path (main_window._on_screenshot),
                # which briefly lifts this app's own capture exclusion — this lock
                # guarantees that window is never overlapped by a live capture tick,
                # so a live frame can never pick up the app's own un-excluded window.
                with window_affinity.capture_affinity_lock:
                    frame = capture.read()
                display_frame = self._mask_exclusion(frame)  # mask only for display (prevents mirror effect)
                detection_frame = frame  # detect on the unmasked frame so faces overlapping the app window aren't lost

                # Emit low-light warning based on a cheap downsampled brightness check.
                thumb = cv2.resize(display_frame, (64, 36))
                gray_mean = float(cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY).mean())
                self.low_light_warning.emit(gray_mean < config.LOW_LIGHT_THRESHOLD)

                with self._focus_lock:
                    focus_rect = self._focus_rect
                if focus_rect is not None:
                    fx, fy, fw, fh = focus_rect
                    hf, wf = detection_frame.shape[:2]
                    x1, y1 = max(0, fx), max(0, fy)
                    x2, y2 = min(wf, fx + fw), min(hf, fy + fh)
                    if x2 > x1 and y2 > y1:
                        detection_frame = detection_frame[y1:y2, x1:x2]
                        fox, foy = x1, y1
                    else:
                        fox, foy = 0, 0
                else:
                    fox, foy = 0, 0

                scale = config.DETECTION_UPSCALE
                if scale != 1.0:
                    h, w = detection_frame.shape[:2]
                    scaled_frame = cv2.resize(
                        detection_frame,
                        (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_CUBIC,
                    )
                    raw_crops = detector.detect(scaled_frame)
                    face_crops = [
                        (crop, {
                            "x": int(b["x"] / scale) + fox,
                            "y": int(b["y"] / scale) + foy,
                            "w": int(b["w"] / scale),
                            "h": int(b["h"] / scale),
                        })
                        for crop, b in raw_crops
                    ]
                else:
                    face_crops = [
                        (crop, {
                            "x": b["x"] + fox,
                            "y": b["y"] + foy,
                            "w": b["w"],
                            "h": b["h"],
                        })
                        for crop, b in detector.detect(detection_frame)
                    ]

                with self._excl_lock:
                    excl = self._excl_rect
                if excl is not None and face_crops:
                    ex, ey, ew, eh = excl
                    face_crops = [
                        (crop, b) for (crop, b) in face_crops
                        if not (ex <= b["x"] + b["w"] // 2 < ex + ew
                                and ey <= b["y"] + b["h"] // 2 < ey + eh)
                    ]

                # Cap to MAX_FACES largest faces so API load stays bounded.
                total_detected = len(face_crops)
                if len(face_crops) > config.MAX_FACES:
                    face_crops = sorted(
                        face_crops, key=lambda fc: fc[1]["w"] * fc[1]["h"], reverse=True
                    )[:config.MAX_FACES]
                self.face_count_update.emit(len(face_crops), total_detected)

                bboxes = [b for _, b in face_crops]
                with self._inflight_lock:
                    inflight_id = self._inflight_id
                # A face mid-HTTP-call must not be evicted just because it
                # left the frame — its response needs a live track to land on.
                protected_ids = frozenset({inflight_id}) if inflight_id else frozenset()
                face_ids = tracker.update(bboxes, protected_ids=protected_ids)
                alive_ids = tracker.alive_ids()  # includes ghosts within their grace window
                keep_ids = alive_ids | protected_ids

                with self._pending_lock:
                    if face_crops:
                        for (crop, bbox), fid in zip(face_crops, face_ids):
                            # Locked faces already have a result — only a
                            # manual re-check (which clears the lock) makes
                            # them eligible for another classification cycle.
                            if fid in self._locked_ids:
                                continue
                            jpeg_bytes, resized_bgr = preprocessor.process_with_array(crop)
                            self._pending_faces[fid] = (jpeg_bytes, bbox, resized_bgr)
                    self._pending_faces = {
                        k: v for k, v in self._pending_faces.items() if k in keep_ids
                    }
                    self._locked_ids &= alive_ids
                    self._recheck_ids &= alive_ids
                with self._result_lock:
                    self._cached_results = {
                        k: v for k, v in self._cached_results.items() if k in keep_ids
                    }

                with self._result_lock, self._pending_lock:
                    results = []
                    pending_bboxes = []
                    rechecking_bboxes = []
                    for fid, bbox in zip(face_ids, bboxes):
                        if fid in self._recheck_ids:
                            rechecking_bboxes.append(bbox)
                        elif fid in self._cached_results:
                            # Refresh to the current bbox so the box keeps
                            # following the face even though it's locked.
                            results.append(replace(self._cached_results[fid], bbox=bbox))
                        else:
                            pending_bboxes.append(bbox)

                self.frame_ready.emit(display_frame, results, pending_bboxes, rechecking_bboxes)

                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.perf_counter() - fps_timer_start
                    fps = 30.0 / elapsed if elapsed > 0 else 0.0
                    self.fps_update.emit(fps)
                    fps_timer_start = time.perf_counter()

                elapsed_frame = time.perf_counter() - t_frame_start
                sleep_time = _FRAME_INTERVAL - elapsed_frame
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            self._running = False
            api_thread.join(timeout=3)
            if owned_engine:
                engine.close()
            if owned_detector:
                detector.close()
            capture.stop()

    def _api_loop(self, engine: DeepfakeInferenceEngine):
        """
        Daemon thread: round-robin through pending face crops, calling the
        BitMind API once per face per cycle (rate limit + retries enforced
        inside the engine). Tracks consecutive failures across cycles so a
        struggling connection surfaces as Degraded rather than crashing the
        thread or silently corrupting cached results.
        """
        consecutive_failures = 0
        degraded = False

        while self._running:
            with self._pending_lock:
                if self._pending_faces:
                    # Take the next face in insertion order
                    face_id, (jpeg_bytes, bbox, resized_bgr) = next(
                        iter(self._pending_faces.items())
                    )
                    # Move it to the back so other faces get a turn next
                    del self._pending_faces[face_id]
                    self._pending_faces[face_id] = (jpeg_bytes, bbox, resized_bgr)
                else:
                    face_id = None

            if face_id is None:
                time.sleep(0.05)
                continue

            with self._inflight_lock:
                self._inflight_id = face_id
            try:
                # Single HTTP attempt, not the multi-attempt retry version —
                # a face stuck retrying internally would block this whole
                # single-threaded loop and starve every other pending face.
                # On failure we just fall through to "continue": the face is
                # already sitting at the back of _pending_faces (moved there
                # above, before this attempt), so it naturally gets retried
                # on its next round-robin turn instead of hogging this one.
                result = engine.run_single_attempt(jpeg_bytes, bbox, face_id)
            except BitmindAuthError as e:
                # An invalid/revoked key never self-heals via retry — escalate immediately.
                consecutive_failures = config.BITMIND_DEGRADED_FAILURE_THRESHOLD
                degraded = self._report_api_failure(e, degraded, consecutive_failures)
                continue
            except BitmindAPIError as e:
                consecutive_failures += 1
                degraded = self._report_api_failure(e, degraded, consecutive_failures)
                continue
            finally:
                with self._inflight_lock:
                    self._inflight_id = None

            if consecutive_failures or degraded:
                consecutive_failures = 0
                degraded = self._report_api_recovery(degraded)

            result.thumbnail = resized_bgr  # exact crop that was analyzed, for Mini mode display

            if result.risk_level == "DEEPFAKE" and config.EXPLANATION_ENABLED:
                result.explanation = analyze_artifacts(resized_bgr)

            with self._result_lock:
                self._cached_results[face_id] = result
            with self._pending_lock:
                was_recheck = face_id in self._recheck_ids
                self._locked_ids.add(face_id)
                self._recheck_ids.discard(face_id)
                # Without this, the entry the pop-and-reinsert above put back
                # into _pending_faces keeps getting round-robin'd forever,
                # even though the face is now locked — the lock only stops
                # NEW insertions, it never removes one already queued.
                self._pending_faces.pop(face_id, None)

            # Logged for every classification, not just alerts, so the raw
            # P_fake distribution can support future FAKE_THRESHOLD recalibration.
            db_log_event(
                "CLASSIFICATION",
                detail=f"face={face_id} risk={result.risk_level} trigger={'manual' if was_recheck else 'auto'}",
                peak_confidence=result.p_fake_raw, api_response_ms=result.api_response_ms,
            )

            self.api_response_update.emit(result.api_response_ms)

    def _report_api_failure(self, error: Exception, was_degraded: bool, consecutive_failures: int) -> bool:
        """Emit/log a Degraded transition once BITMIND_DEGRADED_FAILURE_THRESHOLD consecutive failures are reached."""
        is_degraded = consecutive_failures >= config.BITMIND_DEGRADED_FAILURE_THRESHOLD
        if is_degraded and not was_degraded:
            db_log_event("ERROR", "BitMind API unreachable — detection degraded.")
            self.degraded_mode.emit(True)
        return is_degraded

    def _report_api_recovery(self, was_degraded: bool) -> bool:
        """Emit/log recovery once the API responds successfully again after a Degraded period."""
        if was_degraded:
            db_log_event("INFO", "BitMind API connection recovered.")
            self.degraded_mode.emit(False)
        return False

    def request_recheck(self, face_id: str):
        """Clear a locked face's lock so it runs one more classification cycle."""
        with self._pending_lock:
            self._locked_ids.discard(face_id)
            self._recheck_ids.add(face_id)

    def set_exclusion_region(self, rect: tuple | None):
        """Set a screen region (x, y, w, h) in physical pixels to blank before detection."""
        with self._excl_lock:
            self._excl_rect = rect

    def set_focus_region(self, rect: tuple | None):
        """Restrict detection to a frame-pixel (x, y, w, h) sub-region, or None for full frame."""
        with self._focus_lock:
            self._focus_rect = rect

    def _mask_exclusion(self, frame: np.ndarray) -> np.ndarray:
        """Return frame with the exclusion region zeroed out, or the original if unset."""
        with self._excl_lock:
            excl = self._excl_rect
        if excl is None:
            return frame
        x, y, w, h = excl
        y1 = max(0, y)
        y2 = min(frame.shape[0], y + h)
        x1 = max(0, x)
        x2 = min(frame.shape[1], x + w)
        if y2 <= y1 or x2 <= x1:
            return frame
        masked = frame.copy()
        masked[y1:y2, x1:x2] = 0
        return masked

    def stop(self):
        """Signal the detection loop to stop cleanly."""
        self._running = False

    def set_region(self, region: dict):
        """Update capture region (takes effect on next start)."""
        self._region = region
