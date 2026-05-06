"""
face_crop.py

Usage:
    python face_crop.py --src <frames_dir> --dst <crops_dir>
"""

import os
import argparse
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config

# Download the MediaPipe BlazeFace model file if not present
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[INFO] Downloading MediaPipe BlazeFace model...")
        import urllib.request

        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model downloaded.")


def get_rotation_angle(keypoints, image_width: int, image_height: int) -> float:
    # keypoints[0] = right eye, keypoints[1] = left eye
    rx = keypoints[0].x * image_width
    ry = keypoints[0].y * image_height
    lx = keypoints[1].x * image_width
    ly = keypoints[1].y * image_height
    return math.degrees(math.atan2(ly - ry, lx - rx))


def align_face(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def expand_bbox(x_min, y_min, x_max, y_max, img_w, img_h, ratio):
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


def crop_faces(src_dir: str, dst_dir: str) -> None:
    download_model()
    os.makedirs(dst_dir, exist_ok=True)

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=config.FACE_CONFIDENCE_THRESHOLD,
    )
    detector = mp_vision.FaceDetector.create_from_options(options)

    image_files = [
        f for f in os.listdir(src_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    total = len(image_files)
    saved = 0
    skipped = 0

    for filename in image_files:
        img_path = os.path.join(src_dir, filename)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"[WARNING] Cannot read image: {img_path}. Skipping.")
            skipped += 1
            continue

        h, w = image_bgr.shape[:2]
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        )
        results = detector.detect(mp_image)

        if not results.detections:
            print(f"[WARNING] No face detected: {img_path}. Skipping.")
            skipped += 1
            continue

        det = results.detections[0]
        bb = det.bounding_box
        x_min, y_min = bb.origin_x, bb.origin_y
        x_max = bb.origin_x + bb.width
        y_max = bb.origin_y + bb.height

        x_min, y_min, x_max, y_max = expand_bbox(
            x_min, y_min, x_max, y_max, w, h, config.BBOX_EXPAND_RATIO
        )
        x_min, y_min, x_max, y_max = map(int, (x_min, y_min, x_max, y_max))

        if det.keypoints and len(det.keypoints) >= 2:
            angle = get_rotation_angle(det.keypoints, w, h)
            if abs(angle) > config.ALIGNMENT_ANGLE_THRESHOLD:
                image_bgr = align_face(image_bgr, angle)

        face_crop = image_bgr[y_min:y_max, x_min:x_max]
        if face_crop.size == 0:
            print(f"[WARNING] Empty face crop produced: {img_path}. Skipping.")
            skipped += 1
            continue

        face_crop = cv2.resize(
            face_crop,
            (config.INPUT_SIZE, config.INPUT_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )
        cv2.imwrite(os.path.join(dst_dir, filename), face_crop)
        saved += 1

    detector.close()
    print(f"\n[DONE] Total: {total} | Saved: {saved} | Skipped: {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()
    crop_faces(args.src, args.dst)
