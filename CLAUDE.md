# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Conventions

**All code, filenames, variable names, comments, and identifiers must be written in English.**

## Commands

### Run the application
```bash
python main.py
```
The ONNX model must exist at `models/efficientnet_b4.onnx` before launching. If it is missing, train and export it first (see below).

### Install runtime dependencies
```bash
pip install -r requirements.txt
```

### Install training dependencies
```bash
pip install -r requirements_training.txt
```

### Train the model
```bash
python training/model/train.py --data_dir <crops_root> --label_csv <labels.csv>
```
The label CSV must have columns `filename` (relative path under `data_dir`) and `label` (0 = real, 1 = fake).  
Checkpoints are written to `training/checkpoints/best_model.pt`.

### Export to ONNX (run after training)
```bash
python training/model/export_onnx.py
```
Reads `training/checkpoints/best_model.pt`, writes `models/efficientnet_b4.onnx`.

### Run individual test files
```bash
python test_db.py
python test_history.py
python test_integration.py
```

### Run benchmark
```bash
python benchmark.py
```

## Architecture

The project is split into two independent halves: **runtime** (real-time GUI application) and **training** (offline model preparation). They share a single `config.py` at the repo root.

### Runtime pipeline

The detection pipeline runs entirely inside `DetectionWorker` (a `QThread`) and emits results to the GUI thread via Qt signals:

```
ScreenCapture → FaceDetector → Preprocessor → DeepfakeInferenceEngine → GUI signals
```

1. **`runtime/capture/screen_capture.py`** — Captures the primary monitor using `mss`. Returns BGR `numpy` frames.
2. **`runtime/detection/face_detector.py`** — Detects and aligns faces using MediaPipe BlazeFace (`blaze_face_short_range.tflite`, auto-downloaded on first run). Returns a list of cropped BGR `numpy` arrays per frame.
3. **`runtime/detection/preprocessor.py`** — Normalises crops to `(1, 3, 224, 224)` float32 tensors using ImageNet stats.
4. **`runtime/detection/inference.py`** — Runs EfficientNet-B4 ONNX inference. Maintains per-face EMA state for smoothed display scores. Returns `FaceResult` dataclass objects with `p_fake_raw`, `p_fake_display`, and `risk_level` (GREEN / AMBER / RED).
5. **`runtime/threads/detection_worker.py`** — Orchestrates steps 1–4 in a loop, emits `frame_ready(annotated_frame, results)` and `fps_update(fps)`.
6. **`runtime/threads/alert_thread.py`** — Separate `QThread` that receives `FaceResult` lists via a non-blocking queue and manages debounced alert trigger/dismiss state. Emits `alert_triggered(peak_p_fake)` and `alert_dismissed()`.
7. **`runtime/gui/main_window.py`** — Main PyQt6 window. Wires `DetectionWorker` and `AlertThread` signals to UI slots. Manages session state and history table.
8. **`runtime/storage/database.py`** — SQLite event log (`session_log.db`). Sessions are reconstructed from `SESSION` / `ALERT` / `DISMISS` event rows at read time via `get_all_sessions()`.

### Alert logic (config-driven thresholds)

| Threshold | Constant | Default |
|---|---|---|
| RED alert fires | `FAKE_THRESHOLD` | 0.70 |
| AMBER warning | `WARNING_THRESHOLD` | 0.50 |
| Consecutive RED frames to trigger | `ALERT_TRIGGER_FRAMES` | 2 |
| Consecutive non-RED frames to dismiss | `ALERT_DISMISS_FRAMES` | 5 |

`p_fake_raw` drives alert logic; `p_fake_display` (EMA-smoothed) drives the confidence bar only.

### Training pipeline

`training/preprocessing/` → `training/model/train.py` → `training/model/export_onnx.py`

- **Two-stage fine-tuning**: Stage 1 trains the classifier head only (`STAGE1_EPOCHS = 5`, lr `1e-3`). Stage 2 unfreezes the backbone with differential learning rates (backbone `1e-5`, head `1e-4`).
- **Data split**: Video-level 70 / 15 / 15. The test split is sealed to `training/checkpoints/test_split.csv` at the start of training and must not be used for validation decisions.
- Supported datasets: FaceForensics++ (with C0/C23/C40 compression mixing), Celeb-DF v2, DFDC. Data root paths are set in `config.py`.

### Key configuration (`config.py`)

All runtime and training knobs live here. Notable groups:
- **Paths**: `ONNX_MODEL_PATH`, `DB_PATH`, `CHECKPOINTS_DIR`
- **Inference**: `FAKE_THRESHOLD`, `WARNING_THRESHOLD`, `EMA_ALPHA`
- **Alert debounce**: `ALERT_TRIGGER_FRAMES`, `ALERT_DISMISS_FRAMES`
- **Performance targets**: `TARGET_FPS`, `TARGET_LATENCY_MS`, `TARGET_CPU_PERCENT`, `TARGET_RAM_MB`

### ONNX provider selection

`inference.py:_get_best_providers()` auto-selects providers in priority order: CUDA → DirectML → Azure → CPU. No manual configuration is needed.
