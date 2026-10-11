"""Tests for FaceDetector's pure helpers: aligned cropping, tiling, and duplicate suppression (no model needed)."""

import numpy as np

from runtime.detection.face_detector import _aligned_crop, _suppress_duplicates, _tile_regions

MARKER = (0, 0, 255)


def _frame_with_marker(cx: int, cy: int) -> np.ndarray:
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame[cy - 10:cy + 10, cx - 10:cx + 10] = MARKER
    return frame


def test_aligned_crop_without_rotation_equals_plain_slice():
    frame = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    crop = _aligned_crop(frame, 100, 200, 300, 400, 0.0)
    assert np.array_equal(crop, frame[200:400, 100:300])


def test_aligned_crop_rotates_about_face_not_frame_centre():
    # Face far from the frame centre: rotating about the frame centre would
    # move it hundreds of pixels out of its own bbox.
    frame = _frame_with_marker(200, 150)
    crop = _aligned_crop(frame, 100, 50, 300, 250, 25.0)
    assert crop.shape == (200, 200, 3)
    assert tuple(crop[100, 100]) == MARKER


def test_tile_regions_cover_frame_with_overlap():
    tiles = _tile_regions(1080, 1920, 2, 0.25)
    assert len(tiles) == 4
    for x, y, w, h in tiles:
        assert x >= 0 and y >= 0 and x + w <= 1920 and y + h <= 1080
        assert w > 1920 // 2 and h > 1080 // 2
    assert _tile_regions(1080, 1920, 1, 0.25) == []


def test_suppress_duplicates_merges_overlaps_and_keeps_separate_faces():
    whole_face = (0.9, (100, 100, 200, 200), None)
    half_face_at_tile_edge = (0.5, (100, 100, 150, 200), None)
    other_face = (0.6, (500, 500, 600, 600), None)
    kept = _suppress_duplicates([half_face_at_tile_edge, other_face, whole_face], 0.5)
    assert kept == [whole_face, other_face]
