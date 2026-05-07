"""
face_detector.py

Detects and aligns faces in a BGR frame using the MediaPipe Tasks API.
"""

import os
import math
import urllib.request
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")


def _download_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading MediaPipe BlazeFace model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] BlazeFace model downloaded.")


def _get_rotation_angle(keypoints, img_w: int, img_h: int) -> float:
    rx = keypoints[0].x * img_w
    ry = keypoints[0].y * img_h
    lx = keypoints[1].x * img_w
    ly = keypoints[1].y * img_h
    return math.degrees(math.atan2(ly - ry, lx - rx))


def _align_face(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _expand_bbox(x_min, y_min, x_max, y_max, img_w, img_h, ratio):
    bw = x_max - x_min
    bh = y_max - y_min
    pad_w = int(bw * ratio)
    pad_h = int(bh * ratio)
    return (
        max(0, x_min - pad_w),
        max(0, y_min - pad_h),
        min(img_w, x_max + pad_w),
        min(img_h, y_max + pad_h),
    )


class FaceDetector:
    """
    Detects and aligns faces in BGR frames using MediaPipe BlazeFace (Tasks API).
    """

    def __init__(self):
        _download_model()
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=config.FACE_CONFIDENCE_THRESHOLD,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> list:
        """
        Detect and align all faces in a BGR frame.

        Args:
            frame_bgr: BGR numpy array from screen capture.

        Returns:
            List of (crop, bbox) tuples, one per detected face.
            crop: aligned BGR numpy array; bbox: {"x", "y", "w", "h"} in frame pixels.
            Returns an empty list if no faces are detected.
        """
        h, w = frame_bgr.shape[:2]
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self._detector.detect(mp_image)

        if not results.detections:
            return []

        crops = []
        for det in results.detections:
            bb = det.bounding_box
            x_min = bb.origin_x
            y_min = bb.origin_y
            x_max = bb.origin_x + bb.width
            y_max = bb.origin_y + bb.height

            x_min, y_min, x_max, y_max = _expand_bbox(
                x_min, y_min, x_max, y_max, w, h, config.BBOX_EXPAND_RATIO
            )
            x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))

            frame_work = frame_bgr.copy()
            if det.keypoints and len(det.keypoints) >= 2:
                angle = _get_rotation_angle(det.keypoints, w, h)
                if abs(angle) > config.ALIGNMENT_ANGLE_THRESHOLD:
                    frame_work = _align_face(frame_work, angle)

            crop = frame_work[y_min:y_max, x_min:x_max]
            if crop.size == 0:
                continue

            bbox = {
                "x": x_min,
                "y": y_min,
                "w": x_max - x_min,
                "h": y_max - y_min,
            }
            crops.append((crop, bbox))

        return crops

    def close(self):
        """Release MediaPipe detector resources."""
        self._detector.close()
