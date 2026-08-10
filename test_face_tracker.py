#!/usr/bin/env python3
"""Ad-hoc test for the pure match_tracks()/_iou() functions (no PyQt dependency)."""

from runtime.threads.detection_worker import match_tracks, _iou, _FaceTracker

MAX_DIST = 50
IOU_THRESHOLD = 0.3
GRACE_FRAMES = 5


def box(x, y, w=100, h=100):
    return {"x": x, "y": y, "w": w, "h": h}


# 1. Same position across two frames -> same id.
tracker = _FaceTracker()
ids_1 = tracker.update([box(100, 100)])
ids_2 = tracker.update([box(102, 101)])
assert ids_1 == ids_2, f"expected stable id, got {ids_1} then {ids_2}"
print("[PASS] same-position face keeps the same id")

# 2. Large centroid jump but high IoU overlap -> matched via IoU, same id.
tracker = _FaceTracker()
ids_1 = tracker.update([box(100, 100, 200, 200)])
# Move by 60px (> MAX_DIST=50) but boxes still overlap heavily given the 200x200 size.
ids_2 = tracker.update([box(160, 100, 200, 200)])
assert ids_1 == ids_2, f"expected IoU match despite centroid jump, got {ids_1} then {ids_2}"
print("[PASS] fast movement with high IoU overlap keeps the same id")

# 3. Occlusion for grace_frames - 1 frames, then reappearance -> id retained.
tracker = _FaceTracker()
ids_1 = tracker.update([box(100, 100)])
for _ in range(GRACE_FRAMES - 1):
    tracker.update([])  # face briefly undetected
ids_2 = tracker.update([box(100, 100)])
assert ids_1 == ids_2, f"expected id retained across brief occlusion, got {ids_1} then {ids_2}"
print(f"[PASS] occlusion for {GRACE_FRAMES - 1} frames retains id on reappearance")

# 4. Occlusion beyond grace_frames -> new id assigned.
tracker = _FaceTracker()
ids_1 = tracker.update([box(100, 100)])
for _ in range(GRACE_FRAMES + 1):
    tracker.update([])
ids_2 = tracker.update([box(100, 100)])
assert ids_1 != ids_2, f"expected a new id after grace period expired, got {ids_1} then {ids_2}"
print(f"[PASS] occlusion beyond {GRACE_FRAMES} frames assigns a new id")

# 5. alive_ids() reflects grace-period ghosts, not just the current frame's faces.
tracker = _FaceTracker()
tracker.update([box(100, 100)])
tracker.update([])  # face undetected this frame, but within grace
assert len(tracker.alive_ids()) == 1, f"expected 1 alive id (ghost), got {tracker.alive_ids()}"
print("[PASS] alive_ids() keeps a ghost track alive within its grace window")

# 6. Two independent faces, no crossing -> both tracked with distinct, stable ids.
tracker = _FaceTracker()
ids_1 = tracker.update([box(0, 0), box(500, 500)])
ids_2 = tracker.update([box(5, 5), box(505, 505)])
assert len(set(ids_1)) == 2 and ids_1 == ids_2, f"expected 2 stable distinct ids, got {ids_1} then {ids_2}"
print("[PASS] two independent faces keep distinct, stable ids")

# 7. _iou() sanity checks.
assert _iou(box(0, 0, 100, 100), box(0, 0, 100, 100)) == 1.0
assert _iou(box(0, 0, 100, 100), box(200, 200, 100, 100)) == 0.0
half_overlap = _iou(box(0, 0, 100, 100), box(50, 0, 100, 100))
assert abs(half_overlap - (1 / 3)) < 1e-9, half_overlap
print("[PASS] _iou() identical/disjoint/partial-overlap cases are correct")

# 8. match_tracks() as a pure function directly (no _FaceTracker wrapper).
matched_ids, survivors = match_tracks(
    tracks={"f0": {"bbox": box(100, 100), "missed": 0}},
    bboxes=[box(101, 100)],
    max_dist=MAX_DIST, iou_threshold=IOU_THRESHOLD, grace_frames=GRACE_FRAMES,
)
assert matched_ids == ["f0"] and survivors["f0"]["missed"] == 0
print("[PASS] match_tracks() is independently callable and testable without a QThread")

print("\nKnown limitation (not fixed here): two faces crossing paths can swap ids —")
print("IoU + centroid tracking has no appearance features to disambiguate that case.")
print("\nALL FACE TRACKER TESTS PASSED")
