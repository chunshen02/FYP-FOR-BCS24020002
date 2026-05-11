"""
detection_worker.py

QThread that runs the real-time detection pipeline continuously.
Pipeline: Screen Capture → Face Detection → JPEG Preprocessing → BitMind API
Emits annotated frames and FaceResult lists to the GUI thread.
"""

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
from runtime.detection.inference import DeepfakeInferenceEngine, FaceResult
import config


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
        "GREEN": (0, 255, 0),
        "AMBER": (0, 165, 255),
        "RED":   (0, 0, 255),
    }

    frame_ready = pyqtSignal(object, object)
    fps_update = pyqtSignal(float)
    api_response_update = pyqtSignal(float)
    error_occurred = pyqtSignal(str)

    def __init__(self, region: dict = None, parent=None):
        super().__init__(parent)
        self._region = region
        self._running = False

    def run(self):
        self._running = True

        capture = ScreenCapture(region=self._region)
        detector = FaceDetector()
        preprocessor = Preprocessor()

        try:
            engine = DeepfakeInferenceEngine()
        except EnvironmentError as e:
            self.error_occurred.emit(str(e))
            return

        capture.start()

        frame_count = 0
        fps_timer_start = time.perf_counter()

        try:
            while self._running:
                # Step 1: Capture frame
                frame = capture.read()

                # Step 2: Detect faces
                face_crops = detector.detect(frame)

                # Step 3: Preprocess each crop to JPEG bytes
                jpeg_list = [preprocessor.process(crop) for crop in face_crops]

                # Step 4: Submit to BitMind API
                results = engine.run(jpeg_list) if jpeg_list else []

                # Step 5: Emit API response time if available
                if results:
                    avg_api_ms = sum(r.api_response_ms for r in results) / len(results)
                    self.api_response_update.emit(avg_api_ms)

                # Step 6: Draw overlays
                annotated_frame = self._draw_overlays(frame.copy(), results)

                # Emit to GUI thread
                self.frame_ready.emit(annotated_frame, results)

                # FPS calculation (every 30 frames)
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.perf_counter() - fps_timer_start
                    fps = 30.0 / elapsed if elapsed > 0 else 0.0
                    self.fps_update.emit(fps)
                    fps_timer_start = time.perf_counter()

        except Exception as e:
            self.error_occurred.emit(str(e))
        finally:
            engine.reset_ema()
            engine.close()
            detector.close()
            capture.stop()

    def stop(self):
        """Signal the detection loop to stop cleanly."""
        self._running = False

    def set_region(self, region: dict):
        """Update capture region (takes effect on next start)."""
        self._region = region

    def _draw_overlays(self, frame: np.ndarray, results: list) -> np.ndarray:
        """
        Draw confidence score labels on the frame.
        GREEN < 0.5 | AMBER 0.5–0.7 | RED >= 0.7
        """
        for i, result in enumerate(results):
            colour = self._OVERLAY_COLOURS.get(result.risk_level, (255, 255, 255))
            label = (f"Fake: {result.p_fake_display:.2%} "
                     f"[{result.risk_level}] "
                     f"API: {result.api_response_ms:.0f}ms")
            y_offset = 30 + i * 50
            cv2.putText(frame, label, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        colour, 2, cv2.LINE_AA)
        return frame
