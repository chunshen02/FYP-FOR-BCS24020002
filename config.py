import os

# -- Paths ---------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "training", "data")
FF_PLUS_PLUS_DIR = os.path.join(DATA_DIR, "ff_plus_plus")
CELEB_DF_DIR = os.path.join(DATA_DIR, "celeb_df_v2")
DFDC_DIR = os.path.join(DATA_DIR, "dfdc")

CHECKPOINTS_DIR = os.path.join(BASE_DIR, "training", "checkpoints")
MODELS_DIR = os.path.join(BASE_DIR, "models")
ONNX_MODEL_PATH = os.path.join(MODELS_DIR, "efficientnet_b4.onnx")

DB_PATH = os.path.join(BASE_DIR, "session_log.db")

# -- Model ---------------------------------------------------------------------
MODEL_NAME = "efficientnet_b4"
INPUT_SIZE = 224
INPUT_MEAN = [0.485, 0.456, 0.406]
INPUT_STD = [0.229, 0.224, 0.225]
ONNX_OPSET_VERSION = 17

# -- Training ------------------------------------------------------------------
BATCH_SIZE = 32
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
DROPOUT = 0.4
WEIGHT_DECAY = 1e-4

# Stage 1: head only (epochs 1-5)
STAGE1_EPOCHS = 5
STAGE1_LR = 1e-3

# Stage 2: full fine-tune (epochs 6-30)
STAGE2_BACKBONE_LR = 1e-5
STAGE2_HEAD_LR = 1e-4

# Dataset split ratio (video-level, FF++ only)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Compression-aware training mix (FF++ only)
C40_RATIO = 0.40   # heavily compressed
C23_RATIO = 0.40   # high-bandwidth
C0_RATIO = 0.20    # raw uncompressed

# FFmpeg augmentation
AUG_PROBABILITY = 0.5          # probability of applying FFmpeg compression per batch
AUG_CRF_MIN = 23
AUG_CRF_MAX = 45

# -- Face Detection -------------------------------------------------------------
FACE_CONFIDENCE_THRESHOLD = 0.7
BBOX_EXPAND_RATIO = 0.20       # expand bounding box by 20%
ALIGNMENT_ANGLE_THRESHOLD = 15 # degrees; apply affine alignment if exceeded

# -- Inference -----------------------------------------------------------------
FAKE_THRESHOLD = 0.70          # P_fake >= 0.70 triggers RED alert
WARNING_THRESHOLD = 0.50       # P_fake >= 0.50 triggers AMBER warning
EMA_ALPHA = 0.3                # exponential moving average smoothing factor

# Alert trigger/dismiss rules
ALERT_TRIGGER_FRAMES = 2       # consecutive frames above threshold to trigger alert
ALERT_DISMISS_FRAMES = 5       # consecutive frames below threshold to dismiss alert

# -- Performance Targets --------------------------------------------------------
TARGET_FPS = 10                # minimum acceptable throughput
TARGET_LATENCY_MS = 100        # maximum end-to-end latency
TARGET_CPU_PERCENT = 70        # maximum CPU utilisation
TARGET_RAM_MB = 800            # maximum peak RAM usage

# Benchmark settings
BENCHMARK_WARMUP_SECONDS = 30
BENCHMARK_MEASURE_SECONDS = 60
BENCHMARK_REPEAT = 3

# -- Evaluation ----------------------------------------------------------------
CLASSIFICATION_THRESHOLD = 0.5        # fixed threshold for metric computation
BOOTSTRAP_ITERATIONS = 1000           # for 95% confidence intervals
OVERFITTING_AUC_DROP_LIMIT = 0.15     # max allowed AUC drop on cross-dataset test
