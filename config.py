import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "session_log.db")

# ── BitMind API ────────────────────────────────────────────────────────────────
MODEL_NAME = "bitmind_api"
BITMIND_API_ENDPOINT = "https://api.bitmind.ai/oracle/v1/image-detect"
BITMIND_API_TIMEOUT_SEC = 10          # HTTP request timeout in seconds
BITMIND_RATE_LIMIT_DELAY_SEC = 1.0    # Minimum seconds between API calls (1 req/sec free tier)

# ── Face Detection ─────────────────────────────────────────────────────────────
FACE_CONFIDENCE_THRESHOLD = 0.7       # MediaPipe BlazeFace detection confidence
BBOX_EXPAND_RATIO = 0.20              # Expand bounding box by 20% on all sides
ALIGNMENT_ANGLE_THRESHOLD = 15        # Degrees; apply affine alignment if exceeded

# ── Face Crop Preparation ──────────────────────────────────────────────────────
FACE_CROP_JPEG_QUALITY = 95           # JPEG quality for API submission (0-100)

# ── Inference & Scoring ────────────────────────────────────────────────────────
FAKE_THRESHOLD = 0.70                 # P_fake >= 0.70 triggers RED alert
WARNING_THRESHOLD = 0.50              # P_fake >= 0.50 triggers AMBER warning
EMA_ALPHA = 0.3                       # Exponential moving average smoothing factor
                                      # Smoothed score used for display only.
                                      # Raw score used for alert triggering.

# ── Alert Rules ────────────────────────────────────────────────────────────────
ALERT_TRIGGER_FRAMES = 2              # Consecutive RED frames required to trigger alert
ALERT_DISMISS_FRAMES = 5              # Consecutive non-RED frames required to dismiss alert

# ── Performance Targets ────────────────────────────────────────────────────────
TARGET_FPS = 10                       # Minimum acceptable throughput (frames/sec)
TARGET_LATENCY_MS = 100               # Maximum end-to-end latency (ms)
TARGET_CPU_PERCENT = 70               # Maximum CPU utilisation (%)
TARGET_RAM_MB = 800                   # Maximum peak RAM usage (MB)

# ── Benchmark Settings ─────────────────────────────────────────────────────────
BENCHMARK_WARMUP_SECONDS = 30
BENCHMARK_MEASURE_SECONDS = 60
BENCHMARK_REPEAT = 3
