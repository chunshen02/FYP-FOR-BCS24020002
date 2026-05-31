"""
detection_worker.py

QThread that runs the real-time detection pipeline continuously.
Pipeline: Screen Capture → Face Detection → JPEG Preprocessing → BitMind API
Emits annotated frames and FaceResult lists to the GUI thread.

Architecture:
  Main loop   — capture + face detection at ~15 FPS; uses cached API results.
  API thread  — processes one face per second (rate limit), round-robin across
                all currently visible faces so every participant gets updated.
  _FaceTracker — assigns stable IDs to faces across frames so per-face EMA
                and result caching remain consistent even as faces reorder.
"""

import threading
import time
import numpy as np
import cv2
from PyQt6.QtCore import QThread, pyqtSignal
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from runtime.capture.screen_capture import ScreenCapture
from runtime.detection.face_detector import FaceDetector
from runtime.detection.preprocessor import Preprocessor
from runtime.detection.inference import DeepfakeInferenceEngine
import config

_DISPLAY_FPS = 15
_FRAME_INTERVAL = 1.0 / _DISPLAY_FPS


class _FaceTracker:
    """Assigns stable IDs to faces across frames using bbox centre proximity."""

    _MAX_DIST = 50  # pixels; faces whose centres move ≤ this are the same person

    def __init__(self):
        self._tracks: dict = {}   # face_id -> (cx, cy)
        self._counter: int = 0

    def update(self, bboxes: list) -> list:
        """Match bboxes to existing tracks; assign new IDs to unmatched faces."""
        centers = [(b["x"] + b["w"] // 2, b["y"] + b["h"] // 2) for b in bboxes]
        new_tracks: dict = {}
        used_ids: set = set()
        ids: list = []

        for cx, cy in centers:
            best_id, best_dist = None, self._MAX_DIST
            for tid, (tx, ty) in self._tracks.items():
                if tid in used_ids:
                    continue
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_id = dist, tid
            if best_id is None:
                best_id = f"f{self._counter}"
                self._counter += 1
            used_ids.add(best_id)
            new_tracks[best_id] = (cx, cy)
            ids.append(best_id)

        self._tracks = new_tracks
        return ids


class DetectionWorker(QThread):
    """
    Runs the full deepfake detection pipeline in a background thread.

    Signals:
        frame_ready(np.ndarray, list): Annotated BGR frame + list of FaceResult.
        fps_update(float): Current processing FPS.
        api_response_update(float): Most recent BitMind API response time (ms).
        error_occurred(str): Error message if the pipeline crashes.
    """

    _OVERLAY_COLOURS = {
        "REAL":     (0, 255, 0),
        "DEEPFAKE": (0, 0, 255),
    }

    frame_ready = pyqtSignal(object, object)
    fps_update = pyqtSignal(float)
    api_response_update = pyqtSignal(float)
    error_occurred = pyqtSignal(str)

    def __init__(self, region: dict = None, parent=None):
        super().__init__(parent)
        self._region = region
        self._running = False
        # Shared state between main loop and API thread
        self._pending_lock = threading.Lock()
        self._pending_faces: dict = {}   # face_id -> (jpeg_bytes, bbox)
        self._result_lock = threading.Lock()
        self._cached_results: dict = {}  # face_id -> FaceResult
        self._excl_lock = threading.Lock()
        self._excl_rect: tuple | None = None  # (x, y, w, h) physical pixels
        self._focus_lock = threading.Lock()
        self._focus_rect: tuple | None = None  # (x, y, w, h) frame pixels

    def run(self):
        self._running = True
        self._pending_faces.clear()
        self._cached_results.clear()

        capture = ScreenCapture(region=self._region)
        detector = FaceDetector()
        preprocessor = Preprocessor()
        tracker = _FaceTracker()

        try:
            engine = DeepfakeInferenceEngine()
        except EnvironmentError as e:
            self.error_occurred.emit(str(e))
            return

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

                frame = capture.read()
                detection_frame = self._mask_exclusion(frame)
                display_frame = self._fill_exclusion_display(frame)  # gradient-filled for display; detection uses zeroed frame

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

                bboxes = [b for _, b in face_crops]
                face_ids = tracker.update(bboxes)

                if face_crops:
                    # Update pending crops for the API thread (one entry per face)
                    with self._pending_lock:
                        for (crop, bbox), fid in zip(face_crops, face_ids):
                            self._pending_faces[fid] = (
                                preprocessor.process(crop), bbox
                            )
                    engine.evict_stale_ema(set(face_ids))
                else:
                    with self._pending_lock:
                        self._pending_faces.clear()
                    with self._result_lock:
                        self._cached_results.clear()

                # Build ordered result list matching currently detected faces
                with self._result_lock:
                    results = [
                        self._cached_results[fid]
                        for fid in face_ids
                        if fid in self._cached_results
                    ]

                annotated_frame = self._draw_overlays(display_frame.copy(), results)
                self.frame_ready.emit(annotated_frame, results)

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
            engine.reset_ema()
            engine.close()
            detector.close()
            capture.stop()

    def _api_loop(self, engine: DeepfakeInferenceEngine):
        """
        Daemon thread: round-robin through pending face crops, calling the
        BitMind API once per face per cycle (rate limit enforced inside engine).
        """
        while self._running:
            with self._pending_lock:
                if self._pending_faces:
                    # Take the next face in insertion order
                    face_id, (jpeg_bytes, bbox) = next(
                        iter(self._pending_faces.items())
                    )
                    # Move it to the back so other faces get a turn next
                    del self._pending_faces[face_id]
                    self._pending_faces[face_id] = (jpeg_bytes, bbox)
                else:
                    face_id = None

            if face_id is None:
                time.sleep(0.05)
                continue

            result = engine.run_single(jpeg_bytes, bbox, face_id)

            with self._result_lock:
                self._cached_results[face_id] = result

            self.api_response_update.emit(result.api_response_ms)

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

    def _fill_exclusion_display(self, frame: np.ndarray) -> np.ndarray:
        """Return a copy of frame with the exclusion region blended from its 4 border edges."""
        with self._excl_lock:
            excl = self._excl_rect
        if excl is None:
            return frame
        x, y, w, h = excl
        fh, fw = frame.shape[:2]
        y1, y2 = max(0, y), min(fh, y + h)
        x1, x2 = max(0, x), min(fw, x + w)
        if y2 <= y1 or x2 <= x1:
            return frame

        rh, rw = y2 - y1, x2 - x1

        left_col  = frame[y1:y2, max(0, x1 - 1)].astype(np.float32)       # (rh, 3)
        right_col = frame[y1:y2, min(fw - 1, x2)].astype(np.float32)      # (rh, 3)
        top_row   = frame[max(0, y1 - 1), x1:x2].astype(np.float32)       # (rw, 3)
        bot_row   = frame[min(fh - 1, y2), x1:x2].astype(np.float32)      # (rw, 3)

        t_h = np.linspace(0.0, 1.0, rw, dtype=np.float32)
        horiz = (left_col[:, np.newaxis, :] * (1.0 - t_h)[np.newaxis, :, np.newaxis]
               + right_col[:, np.newaxis, :] * t_h[np.newaxis, :, np.newaxis])

        t_v = np.linspace(0.0, 1.0, rh, dtype=np.float32)
        vert = (top_row[np.newaxis, :, :] * (1.0 - t_v)[:, np.newaxis, np.newaxis]
              + bot_row[np.newaxis, :, :] * t_v[:, np.newaxis, np.newaxis])

        result = frame.copy()
        result[y1:y2, x1:x2] = np.clip((horiz + vert) * 0.5, 0, 255).astype(np.uint8)
        return result

    def stop(self):
        """Signal the detection loop to stop cleanly."""
        self._running = False

    def set_region(self, region: dict):
        """Update capture region (takes effect on next start)."""
        self._region = region

    def _draw_overlays(self, frame: np.ndarray, results: list) -> np.ndarray:
        """Draw bounding boxes and confidence score labels. REAL < 0.7 | DEEPFAKE >= 0.7"""
        for i, result in enumerate(results):
            colour = self._OVERLAY_COLOURS.get(result.risk_level, (255, 255, 255))
            b = result.bbox
            x1, y1 = b["x"], b["y"]
            x2, y2 = x1 + b["w"], y1 + b["h"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

            elapsed = time.perf_counter() - result.last_updated
            age_str = "just now" if elapsed < 1.0 else f"{elapsed:.0f}s ago"
            label = (f"{result.p_fake_display:.1%} Deepfake "
                     f"(updated {age_str})")
            label_y = max(y1 - 8, 20)
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        colour, 2, cv2.LINE_AA)
        return frame
