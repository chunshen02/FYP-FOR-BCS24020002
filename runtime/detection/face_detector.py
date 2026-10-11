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


def _align_face(image: np.ndarray, box: tuple, angle: float) -> np.ndarray:
    """Rotate the (x_min, y_min, x_max, y_max) face region upright about its own centre and return it."""
    x_min, y_min, x_max, y_max = box
    # Rotating about the frame centre instead would carry an off-centre face out of its own box.
    mat = cv2.getRotationMatrix2D(((x_min + x_max) / 2, (y_min + y_max) / 2), angle, 1.0)
    # Shift the output origin to the box corner so only the crop region is computed, not the whole frame.
    mat[:, 2] -= (x_min, y_min)
    return cv2.warpAffine(
        image, mat, (x_max - x_min, y_max - y_min), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
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
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def detect(self, frame_bgr: np.ndarray) -> list:
        """
        Detect and align all faces in a BGR frame, detecting on a copy downscaled to DETECTION_MAX_WIDTH.

        Args:
            frame_bgr: BGR numpy array from screen capture.

        Returns:
            List of (crop, bbox) tuples, one per detected face, largest first.
            crop: aligned BGR numpy array cut from frame_bgr at full resolution;
            bbox: {"x", "y", "w", "h"} in frame_bgr pixels.
            Returns an empty list if no faces are detected.
        """
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, config.DETECTION_MAX_WIDTH / w)
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame_bgr

        # CLAHE on the L channel improves face detection in low-light conditions.
        # Crops are still taken from the original frame to preserve image fidelity for the API.
        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        detection_bgr = cv2.cvtColor(
            cv2.merge([self._clahe.apply(l_ch), a_ch, b_ch]), cv2.COLOR_LAB2BGR
        )
        rgb_frame = cv2.cvtColor(detection_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self._detector.detect(mp_image)

        if not results.detections:
            return []

        crops = []
        for det in results.detections:
            bb = det.bounding_box
            x_min, y_min, x_max, y_max = _expand_bbox(
                bb.origin_x / scale, bb.origin_y / scale,
                (bb.origin_x + bb.width) / scale, (bb.origin_y + bb.height) / scale,
                w, h, config.BBOX_EXPAND_RATIO,
            )
            x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))
            if x_max <= x_min or y_max <= y_min:
                continue

            # Keypoints are normalised, so the angle is the same at either resolution.
            angle = _get_rotation_angle(det.keypoints, w, h) if det.keypoints and len(det.keypoints) >= 2 else 0.0
            if abs(angle) > config.ALIGNMENT_ANGLE_THRESHOLD:
                crop = _align_face(frame_bgr, (x_min, y_min, x_max, y_max), angle)
            else:
                crop = frame_bgr[y_min:y_max, x_min:x_max].copy()

            crops.append((crop, {"x": x_min, "y": y_min, "w": x_max - x_min, "h": y_max - y_min}))

        return sorted(crops, key=lambda fc: fc[1]["w"] * fc[1]["h"], reverse=True)

    def close(self):
        """Release MediaPipe detector resources."""
        self._detector.close()
