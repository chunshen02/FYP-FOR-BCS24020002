import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "session_log.db")

# ── BitMind API ────────────────────────────────────────────────────────────────
MODEL_NAME = "bitmind_api"
BITMIND_API_ENDPOINT = "https://api.bitmind.ai/detect-image"
BITMIND_API_TIMEOUT_SEC = 10          # HTTP request timeout in seconds
BITMIND_RATE_LIMIT_DELAY_SEC = 1.0    # Minimum seconds between API calls (1 req/sec free tier)
BITMIND_RATE_LIMIT_BACKOFF_SEC = 5.0       # First pause after HTTP 429 when no Retry-After header; doubles per repeat
BITMIND_RATE_LIMIT_BACKOFF_MAX_SEC = 60.0  # Upper bound for the HTTP 429 pause
BITMIND_DEGRADED_FAILURE_THRESHOLD = 3       # Consecutive transient failures before Degraded status

# ── Face Detection ─────────────────────────────────────────────────────────────
FACE_CONFIDENCE_THRESHOLD = 0.3       # MediaPipe BlazeFace detection confidence. Raised from 0.2: on FF++
                                       # frames it removes ~60% of extra (mostly non-face) boxes, which would
                                       # otherwise use API calls and risk false alerts, for a 0.2-6% per-frame
                                       # recall cost that the tracker's grace period absorbs.
# Boxes between FACE_CONFIDENCE_THRESHOLD and this score are tracked but never drawn, counted or sent
# to BitMind. On FF++ clips shown full screen, real faces scored 0.9+ while the non-face boxes
# (background, microphones) that 0.3 lets through peaked at 0.70 and took ~65% of BitMind calls;
# 0.6 kept all but 9 of 273 such tracks out, yet still accepted every face of a simulated 3x3
# call gallery within 1 s.
FACE_ACCEPT_SCORE = 0.6               # A track counts as a face once one of its detections scores at least this
BBOX_EXPAND_RATIO = 0.20              # Expand bounding box by 20% on all sides
ALIGNMENT_ANGLE_THRESHOLD = 15        # Degrees; apply affine alignment if exceeded
# BlazeFace downsamples every input to 128x128, so upscaling the frame never
# helps small faces — running it on overlapping sub-tiles does (each tile is
# downsampled less). The full frame is always run too, for faces too large
# to fit inside a single tile.
DETECTION_MAX_WIDTH = 1280            # Wider frames are downscaled before detection (cheaper CLAHE/colour conversion;
                                      # tiles stay far above BlazeFace's 128x128 input, so nothing is lost)
DETECTION_TILE_GRID = 3              # N x N overlapping tiles per frame; 1 disables tiling. 3, not 2: on a
                                      # 2560x1600 screen a 3x3 call gallery went from 15% to 100% of faces found
                                      # (4x4: 0% -> 43%) for ~5 ms more detection per frame.
DETECTION_TILE_OVERLAP = 0.25         # Each tile extends this fraction past its grid cell
DETECTION_DUPLICATE_OVERLAP = 0.5     # Intersection / smaller-box area above which two detections are one face
MAX_PENDING_FACES = 3                 # Faces awaiting a BitMind verdict at once (most confident first). Every
                                       # accepted face is tracked and drawn; a classified face frees its slot
                                       # for the next one, so crowded meetings are worked through in turn.

# ── Multi-Face Tracking ────────────────────────────────────────────────────────
FACE_TRACK_MAX_DIST_PX = 50           # Centroid-distance fallback threshold (pixels) for matching
FACE_TRACK_IOU_THRESHOLD = 0.3        # Min IoU to consider two boxes across frames the same face
FACE_TRACK_GRACE_FRAMES = 5           # Frames a track survives with no matching bbox (~333ms at DISPLAY_FPS=15)

# ── Face Crop Preparation ──────────────────────────────────────────────────────
FACE_CROP_JPEG_QUALITY = 85           # JPEG quality for API submission (0-100)
FACE_CROP_SIZE = 224                  # Resize face crop to this square size before API submission

# ── Inference & Scoring ────────────────────────────────────────────────────────
# FAKE_THRESHOLD was calibrated in FYP1 on a small sample (n=3 deepfake, n=10
# real). Every classification is now
# logged with its raw P_fake score (CLASSIFICATION events in session_log.db) so
# this can be recalibrated against a larger, more diverse sample in FYP2.
FAKE_THRESHOLD = 0.70                 # P_fake >= 0.70 → DEEPFAKE; below → REAL (adjustable in Settings)
# A single BitMind verdict is noisy, and a cached result repeats every frame,
# so frame-based alert debouncing alone would let one misjudged call raise an
# alert. A first REAL verdict is final (one call); a first DEEPFAKE verdict is
# re-checked on fresh frames until CONFIRM_FAKE_VOTES of CONFIRM_VERDICTS agree
# either way. The face's final score is the median of its verdicts.
CONFIRM_VERDICTS = 3
CONFIRM_FAKE_VOTES = 2

# ── Alert Rules ────────────────────────────────────────────────────────────────
ALERT_TRIGGER_FRAMES = 3              # Consecutive RED frames required to trigger alert
ALERT_DISMISS_FRAMES = 8              # Consecutive non-RED frames required to dismiss alert
ALERT_SOUND_ENABLED = True            # Beep on alert (adjustable in Settings)
ALERT_NOTIFICATION_ENABLED = True     # Desktop toast on alert (adjustable in Settings)
ALERT_AUTO_SCREENSHOT = True          # Save a full-screen evidence PNG on alert (adjustable in Settings)

# ── Display / Loop Rate ────────────────────────────────────────────────────────
DISPLAY_FPS = 15                      # Detection loop tick rate (frames/sec)

# ── Low-Light Handling ────────────────────────────────────────────────────────
LOW_LIGHT_THRESHOLD = 40              # Frame mean brightness (0-255) below which low-light mode activates
LOW_LIGHT_GAMMA = 0.5                 # Gamma exponent applied to dark face crops before API submission

# ── Local Explainability (CV Heuristics) ────────────────────────────────────────
# Thresholds calibrated against test_samples/ (10 real + 10 deepfake crops).
# Note: on that small sample, none of the three metrics separated real from
# deepfake — set instead to flag statistical outliers relative to the observed
# range of ordinary face crops (roughly pooled mean + 2 std), so they surface
# genuinely unusual crops rather than firing on every face. Treat as
# corroborating, best-effort evidence, not a calibrated classifier.
EXPLANATION_ENABLED = True             # Run local CV artifact heuristics on DEEPFAKE-classified faces
EXPLAIN_FFT_HF_RATIO_THRESHOLD = 0.0004  # High/low FFT energy ratio above which spectral anomaly is flagged
EXPLAIN_NOISE_RATIO_THRESHOLD = 2.6      # Inner/border noise-residual variance ratio above which blending is flagged
EXPLAIN_ASYMMETRY_THRESHOLD = 0.20       # Normalized left-right pixel-diff mean above which asymmetry is flagged

# ── Evaluation Scripts (detection_accuracy.py / realtime_performance.py / screen_face_pipeline.py) ──
FF_DATASET_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "archive", "FaceForensics++_C23"))
FF_REAL_SUBSET = "original"
# Each fake "<target>_<source>.mp4" is built from original/<target>.mp4, so the
# first N videos of every subset show the same scenes.
FF_FAKE_SUBSETS = ["Deepfakes", "Face2Face", "FaceSwap", "FaceShifter", "NeuralTextures"]
RESULTS_DIR = os.path.join(BASE_DIR, "results")
ACCURACY_VIDEOS_PER_SUBSET = 100      # First N videos (by name) of each subset
PERF_VIDEOS_PER_SUBSET = 4            # Videos per subset played during the performance test
PIPELINE_VIDEOS_PER_SUBSET = 2        # Videos per subset shown full screen during the screen face pipeline test
PERF_WARMUP_SECONDS = 10              # Excluded from the performance and pipeline tests' metrics
TARGET_FPS = 10                       # Minimum acceptable main-loop throughput (frames/sec)
TARGET_FRAME_LATENCY_MS = 100         # Maximum mean per-frame latency of capture + detection + tracking (ms)
# A DEEPFAKE verdict takes 2-3 sequential BitMind calls (CONFIRM_VERDICTS), and BitMind's response
# time varies by day (1.8-5.1 s means seen; median 4.1 s over 984 calls on 2026-10-13), so three
# calls alone take ~12 s. The end-to-end target allows that plus some queueing; the app-added target
# holds the part the app controls, kept under one median call so a face never waits a whole extra call.
TARGET_DETECTION_LATENCY_MS = 15000   # Maximum mean time from a face appearing to its final verdict (ms)
TARGET_APP_LATENCY_MS = 3000          # Maximum mean part of that time spent outside the face's own BitMind requests (ms)
TARGET_CPU_PERCENT = 70               # Maximum mean CPU use of the app process (% of all cores)
TARGET_RAM_MB = 800                   # Maximum peak RAM of the app process (MB)
TARGET_FACE_HIT_RATE = 0.90           # Minimum share of on-screen clip frames whose screen grab yields a face crop
TARGET_TRACK_CONTINUITY = 0.95        # Minimum share of a clip's face frames that keep its main face's track ID
