"""Tests for the pure match_tracks()/_iou() functions and _FaceTracker (no QApplication or QThread needed)."""

import config
from runtime.threads.detection_worker import match_tracks, _iou, _FaceTracker

MAX_DIST = config.FACE_TRACK_MAX_DIST_PX
IOU_THRESHOLD = config.FACE_TRACK_IOU_THRESHOLD
GRACE_FRAMES = config.FACE_TRACK_GRACE_FRAMES
# Centroid jump just past MAX_DIST — still a heavy IoU overlap for 200x200 boxes.
FAST_JUMP = MAX_DIST + 10


def box(x, y, w=100, h=100):
    return {"x": x, "y": y, "w": w, "h": h}


def test_same_position_keeps_id():
    tracker = _FaceTracker()
    ids_1 = tracker.update([box(100, 100)])
    ids_2 = tracker.update([box(102, 101)])
    assert ids_1 == ids_2


def test_fast_movement_with_high_iou_keeps_id():
    tracker = _FaceTracker()
    ids_1 = tracker.update([box(100, 100, 200, 200)])
    ids_2 = tracker.update([box(100 + FAST_JUMP, 100, 200, 200)])
    assert ids_1 == ids_2


def test_brief_occlusion_retains_id():
    tracker = _FaceTracker()
    ids_1 = tracker.update([box(100, 100)])
    for _ in range(GRACE_FRAMES):
        tracker.update([])
    ids_2 = tracker.update([box(100, 100)])
    assert ids_1 == ids_2


def test_occlusion_beyond_grace_assigns_new_id():
    tracker = _FaceTracker()
    ids_1 = tracker.update([box(100, 100)])
    for _ in range(GRACE_FRAMES + 1):
        tracker.update([])
    ids_2 = tracker.update([box(100, 100)])
    assert ids_1 != ids_2


def test_alive_ids_includes_ghost_within_grace():
    tracker = _FaceTracker()
    tracker.update([box(100, 100)])
    tracker.update([])
    assert len(tracker.alive_ids()) == 1


def test_two_independent_faces_keep_distinct_stable_ids():
    tracker = _FaceTracker()
    ids_1 = tracker.update([box(0, 0), box(500, 500)])
    ids_2 = tracker.update([box(5, 5), box(505, 505)])
    assert len(set(ids_1)) == 2
    assert ids_1 == ids_2


def test_iou_identical_disjoint_and_partial():
    assert _iou(box(0, 0), box(0, 0)) == 1.0
    assert _iou(box(0, 0), box(200, 200)) == 0.0
    assert abs(_iou(box(0, 0), box(50, 0)) - 1 / 3) < 1e-9


def test_match_tracks_callable_without_qthread():
    matched_ids, survivors = match_tracks(
        tracks={"f0": {"bbox": box(100, 100), "missed": 0}},
        bboxes=[box(101, 100)],
        max_dist=MAX_DIST, iou_threshold=IOU_THRESHOLD, grace_frames=GRACE_FRAMES,
    )
    assert matched_ids == ["f0"]
    assert survivors["f0"]["missed"] == 0


def test_iou_rescues_id_that_centroid_only_would_lose():
    tracks_before = {"f0": {"bbox": box(100, 100, 200, 200), "missed": 0}}
    moved_bbox = [box(100 + FAST_JUMP, 100, 200, 200)]

    matched_with_iou, _ = match_tracks(tracks_before, moved_bbox, MAX_DIST, IOU_THRESHOLD, GRACE_FRAMES)
    # iou_threshold=1.1 is unreachable (IoU is bounded at 1.0), forcing pure centroid fallback.
    matched_centroid_only, _ = match_tracks(tracks_before, moved_bbox, MAX_DIST, 1.1, GRACE_FRAMES)

    assert matched_with_iou == ["f0"]
    assert matched_centroid_only == [None]


def test_single_frame_position_swap_reassigns_ids_by_position():
    # Confirmed limitation, not a bug: no position-only signal (centroid or IoU)
    # can tell "these two identities crossed" from "two faces stayed put".
    tracker = _FaceTracker()
    left_id, right_id = tracker.update([box(0, 0), box(300, 0)])
    id_at_300, id_at_0 = tracker.update([box(300, 0), box(0, 0)])
    assert id_at_300 == right_id
    assert id_at_0 == left_id
