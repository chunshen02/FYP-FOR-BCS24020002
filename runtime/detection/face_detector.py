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


<<<<<<< HEAD
def _aligned_crop(image: np.ndarray, x_min: int, y_min: int, x_max: int, y_max: int,
                  angle: float) -> np.ndarray:
    """Rotate the face upright about its own centre and return only its bbox region."""
    centre = ((x_min + x_max) / 2, (y_min + y_max) / 2)
    mat = cv2.getRotationMatrix2D(centre, angle, 1.0)
    # Shifting the bbox's top-left to the output origin makes warpAffine
    # compute only the crop's pixels instead of rotating the whole frame.
    mat[0, 2] -= x_min
    mat[1, 2] -= y_min
    return cv2.warpAffine(
        image, mat, (x_max - x_min, y_max - y_min),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
=======
def _align_face(image: np.ndarray, box: tuple, angle: float) -> np.ndarray:
    """Rotate the (x_min, y_min, x_max, y_max) face region upright about its own centre and return it."""
    x_min, y_min, x_max, y_max = box
    # Rotating about the frame centre instead would carry an off-centre face out of its own box.
    mat = cv2.getRotationMatrix2D(((x_min + x_max) / 2, (y_min + y_max) / 2), angle, 1.0)
    # Shift the output origin to the box corner so only the crop region is computed, not the whole frame.
    mat[:, 2] -= (x_min, y_min)
    return cv2.warpAffine(
        image, mat, (x_max - x_min, y_max - y_min), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
>>>>>>> 3bef4a09fe8c51c26d29703668d4889ad5fe4d31
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


def _tile_regions(img_h: int, img_w: int, grid: int, overlap: float) -> list:
    """Return (x, y, w, h) for a grid x grid layout of tiles, each extended by overlap past its cell."""
    if grid <= 1:
        return []
    tile_h = min(img_h, int(img_h / grid * (1 + overlap)))
    tile_w = min(img_w, int(img_w / grid * (1 + overlap)))
    return [
        (min(img_w - tile_w, int(col * img_w / grid)), min(img_h - tile_h, int(row * img_h / grid)), tile_w, tile_h)
        for row in range(grid)
        for col in range(grid)
    ]


def _overlap_ratio(a: tuple, b: tuple) -> float:
    """Intersection area over the smaller box's area, for (x_min, y_min, x_max, y_max) boxes."""
    inter_w = min(a[2], b[2]) - max(a[0], b[0])
    inter_h = min(a[3], b[3]) - max(a[1], b[1])
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter_w * inter_h / smaller


def _suppress_duplicates(detections: list, threshold: float) -> list:
    """
    Keep detections (score, box, angle) highest-score first, dropping any
    that overlap an already-kept box. Overlap is measured against the
    smaller box so a half-face cut off at a tile edge is suppressed by the
    whole face found in the neighbouring tile, even though their IoU is low.
    """
    kept = []
    for det in sorted(detections, key=lambda d: d[0], reverse=True):
        if all(_overlap_ratio(det[1], k[1]) <= threshold for k in kept):
            kept.append(det)
    return kept


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
<<<<<<< HEAD
            List of (crop, bbox) tuples, one per detected face, most confident first.
            crop: aligned BGR numpy array; bbox: {"x", "y", "w", "h"} in frame pixels.
=======
            List of (crop, bbox) tuples, one per detected face, largest first.
            crop: aligned BGR numpy array cut from frame_bgr at full resolution;
            bbox: {"x", "y", "w", "h"} in frame_bgr pixels.
>>>>>>> 3bef4a09fe8c51c26d29703668d4889ad5fe4d31
            Returns an empty list if no faces are detected.
        """
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, config.DETECTION_MAX_WIDTH / w)
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else frame_bgr

<<<<<<< HEAD
        # Detection runs on a downscaled copy; crops are still taken from the
        # original frame to preserve image fidelity for the API.
        scale = min(1.0, config.DETECTION_MAX_WIDTH / w)
        if scale < 1.0:
            detection_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            detection_bgr = frame_bgr
        det_h, det_w = detection_bgr.shape[:2]
=======
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
>>>>>>> 3bef4a09fe8c51c26d29703668d4889ad5fe4d31

        # CLAHE on the L channel improves face detection in low light; skip its
        # colour conversions when the frame is bright.
        brightness = cv2.cvtColor(cv2.resize(detection_bgr, (64, 36)), cv2.COLOR_BGR2GRAY).mean()
        if brightness < config.LOW_LIGHT_THRESHOLD:
            l_ch, a_ch, b_ch = cv2.split(cv2.cvtColor(detection_bgr, cv2.COLOR_BGR2LAB))
            detection_bgr = cv2.cvtColor(
                cv2.merge([self._clahe.apply(l_ch), a_ch, b_ch]), cv2.COLOR_LAB2BGR
            )
        rgb_frame = cv2.cvtColor(detection_bgr, cv2.COLOR_BGR2RGB)

        detections = self._detect_region(rgb_frame, 0, 0)
        for x, y, tile_w, tile_h in _tile_regions(det_h, det_w, config.DETECTION_TILE_GRID, config.DETECTION_TILE_OVERLAP):
            tile = np.ascontiguousarray(rgb_frame[y:y + tile_h, x:x + tile_w])
            detections += self._detect_region(tile, x, y)

        crops = []
<<<<<<< HEAD
        for _, box, angle in _suppress_duplicates(detections, config.DETECTION_DUPLICATE_OVERLAP):
            box = tuple(int(v / scale) for v in box)
            x_min, y_min, x_max, y_max = _expand_bbox(*box, w, h, config.BBOX_EXPAND_RATIO)
            if x_max <= x_min or y_max <= y_min:
                continue

            if angle is not None and abs(angle) > config.ALIGNMENT_ANGLE_THRESHOLD:
                crop = _aligned_crop(frame_bgr, x_min, y_min, x_max, y_max, angle)
            else:
                crop = frame_bgr[y_min:y_max, x_min:x_max].copy()

            bbox = {
                "x": x_min,
                "y": y_min,
                "w": x_max - x_min,
                "h": y_max - y_min,
            }
            crops.append((crop, bbox))

        return crops
=======
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
>>>>>>> 3bef4a09fe8c51c26d29703668d4889ad5fe4d31

    def _detect_region(self, rgb: np.ndarray, x_offset: int, y_offset: int) -> list:
        """Run BlazeFace on an RGB image; return (score, box offset into the detection frame, roll angle or None) per face."""
        img_h, img_w = rgb.shape[:2]
        results = self._detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        detections = []
        for det in results.detections:
            bb = det.bounding_box
            box = (
                bb.origin_x + x_offset,
                bb.origin_y + y_offset,
                bb.origin_x + bb.width + x_offset,
                bb.origin_y + bb.height + y_offset,
            )
            angle = (
                _get_rotation_angle(det.keypoints, img_w, img_h)
                if det.keypoints and len(det.keypoints) >= 2 else None
            )
            detections.append((det.categories[0].score, box, angle))
        return detections

    def close(self):
        """Release MediaPipe detector resources."""
        self._detector.close()
