"""
detection_worker.py

QThread that runs the real-time detection pipeline continuously.
Emits raw frames and FaceResult lists (with bboxes) to the GUI thread.
"""

import time
import numpy as np
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
        frame_ready(np.ndarray, list): Raw BGR frame + list of FaceResult with bboxes.
        fps_update(float): Current processing FPS.
        error_occurred(str): Error message if the pipeline crashes.
    """

    frame_ready = pyqtSignal(object, object)  # (frame: np.ndarray, results: list[FaceResult])
    fps_update = pyqtSignal(float)
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
        engine = DeepfakeInferenceEngine()

        capture.start()

        frame_count = 0
        fps_timer_start = time.perf_counter()

        try:
            while self._running:
                # Step 1: Capture frame
                frame = capture.read()

                # Step 2: Detect faces — returns list of (crop, bbox) tuples
                detections = detector.detect(frame)
                face_crops = [d[0] for d in detections]
                bboxes = [d[1] for d in detections]

                # Step 3: Preprocess each crop
                tensors = [preprocessor.process(crop) for crop in face_crops]

                # Step 4: Run inference
                results = engine.run(tensors) if tensors else []

                # Step 5: Attach MediaPipe bboxes directly to results
                for result, bbox in zip(results, bboxes):
                    result.bbox = bbox

                # Emit raw frame + results; VideoWidget handles all rendering
                self.frame_ready.emit(frame, results)

                # FPS calculation (update every 30 frames)
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
            detector.close()
            capture.stop()

    def stop(self):
        """Signal the detection loop to stop cleanly."""
        self._running = False

    def set_region(self, region: dict):
        """Update capture region (takes effect on next start)."""
        self._region = region
