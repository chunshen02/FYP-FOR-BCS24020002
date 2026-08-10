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

# 9. Head-to-head proof that IoU changes the outcome vs. centroid-only fallback,
# on the exact same fast-movement input as test 2 (60px jump > MAX_DIST=50px).
tracks_before = {"f0": {"bbox": box(100, 100, 200, 200), "missed": 0}}
moved_bbox = [box(160, 100, 200, 200)]

matched_with_iou, _ = match_tracks(tracks_before, moved_bbox, MAX_DIST, IOU_THRESHOLD, GRACE_FRAMES)
# iou_threshold=1.1 is unreachable (IoU is bounded at 1.0), forcing pure centroid fallback.
matched_centroid_only, _ = match_tracks(tracks_before, moved_bbox, MAX_DIST, 1.1, GRACE_FRAMES)

assert matched_with_iou == ["f0"], f"expected IoU path to keep the id, got {matched_with_iou}"
assert matched_centroid_only == [None], (
    f"expected centroid-only fallback (IoU disabled) to LOSE the id on this same input "
    f"(60px jump > MAX_DIST={MAX_DIST}px), got {matched_centroid_only} — if this fails, "
    f"IoU isn't actually changing the outcome versus centroid-only matching"
)
print("[PASS] head-to-head comparison: IoU rescues an id that centroid-only matching would lose")

# 10. Two faces fully swapping screen positions in a single frame — CONFIRMED,
# characterized limitation, not silently assumed. No position-only signal (centroid
# or IoU) can tell "these two identities crossed" apart from "two faces stayed put";
# the swapped detection lands exactly on the OTHER track's last known position, so
# it inherits that track's id instead of its own physical owner's.
tracker = _FaceTracker()
ids_before = tracker.update([box(0, 0), box(300, 0)])   # left id, right id
left_id_before, right_id_before = ids_before
ids_after = tracker.update([box(300, 0), box(0, 0)])    # physically swapped in one frame
matched_at_300, matched_at_0 = ids_after

assert matched_at_300 == right_id_before, "expected the id that WAS at (300,0) to stay assigned to (300,0)"
assert matched_at_0 == left_id_before, "expected the id that WAS at (0,0) to stay assigned to (0,0)"
print("[PASS] confirmed limitation: a full single-frame position swap reassigns ids by")
print("       position, not by which physical face actually moved — inherent to any")
print("       position-only tracker (centroid or IoU alike), not something IoU fixes.")
print("\nALL FACE TRACKER TESTS PASSED")
