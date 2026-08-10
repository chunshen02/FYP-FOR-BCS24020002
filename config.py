import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "session_log.db")

# ── BitMind API ────────────────────────────────────────────────────────────────
MODEL_NAME = "bitmind_api"
BITMIND_API_ENDPOINT = "https://api.bitmind.ai/detect-image"
BITMIND_API_TIMEOUT_SEC = 10          # HTTP request timeout in seconds
BITMIND_RATE_LIMIT_DELAY_SEC = 1.0    # Minimum seconds between API calls (1 req/sec free tier)
BITMIND_RETRY_DELAYS_SEC = [1.0, 2.0, 4.0]   # Exponential backoff for transient (timeout/conn/5xx) failures
BITMIND_DEGRADED_FAILURE_THRESHOLD = 3       # Consecutive failures before Degraded status; auth errors bypass this

# ── Face Detection ─────────────────────────────────────────────────────────────
FACE_CONFIDENCE_THRESHOLD = 0.2       # MediaPipe BlazeFace detection confidence
BBOX_EXPAND_RATIO = 0.20              # Expand bounding box by 20% on all sides
ALIGNMENT_ANGLE_THRESHOLD = 15        # Degrees; apply affine alignment if exceeded
DETECTION_UPSCALE = 2.0               # Upscale factor applied to frame before BlazeFace (helps small faces)
MAX_FACES = 3                         # Analyze at most this many faces per frame (largest first)

# ── Multi-Face Tracking ────────────────────────────────────────────────────────
FACE_TRACK_MAX_DIST_PX = 50           # Centroid-distance fallback threshold (pixels) for matching
FACE_TRACK_IOU_THRESHOLD = 0.3        # Min IoU to consider two boxes across frames the same face
FACE_TRACK_GRACE_FRAMES = 5           # Frames a track survives with no matching bbox (~333ms at DISPLAY_FPS=15)

# ── Face Crop Preparation ──────────────────────────────────────────────────────
FACE_CROP_JPEG_QUALITY = 85           # JPEG quality for API submission (0-100)
FACE_CROP_SIZE = 224                  # Resize face crop to this square size before API submission

# ── Inference & Scoring ────────────────────────────────────────────────────────
# FAKE_THRESHOLD was calibrated in FYP1 on a small sample (n=3 deepfake, n=10
# real) — see PERFORMANCE_RESULTS.txt Table 4/5. Every classification is now
# logged with its raw P_fake score (see database.get_score_distribution) so
# this can be recalibrated against a larger, more diverse sample in FYP2.
FAKE_THRESHOLD = 0.70                 # P_fake >= 0.70 → DEEPFAKE; below → REAL
EMA_ALPHA = 0.2                       # Exponential moving average smoothing factor
                                      # Smoothed score used for display only.
                                      # Raw score used for alert triggering.

# ── Alert Rules ────────────────────────────────────────────────────────────────
ALERT_TRIGGER_FRAMES = 3              # Consecutive RED frames required to trigger alert
ALERT_DISMISS_FRAMES = 8              # Consecutive non-RED frames required to dismiss alert

# ── Display / Loop Rate ────────────────────────────────────────────────────────
DISPLAY_FPS = 15                      # Detection loop tick rate (frames/sec)

# ── Performance Targets ────────────────────────────────────────────────────────
TARGET_FPS = 10                       # Minimum acceptable throughput (frames/sec)
TARGET_LATENCY_MS = 100               # Maximum end-to-end latency (ms)
TARGET_CPU_PERCENT = 70               # Maximum CPU utilisation (%)
TARGET_RAM_MB = 800                   # Maximum peak RAM usage (MB)

# ── Benchmark Settings ─────────────────────────────────────────────────────────
BENCHMARK_WARMUP_SECONDS = 30
BENCHMARK_MEASURE_SECONDS = 60
BENCHMARK_REPEAT = 3

# ── Offline Batch Evaluation (evaluation_runner.py) ─────────────────────────────
EVAL_FRAMES_PER_VIDEO = 5             # Frames sampled per video clip during batch evaluation

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
