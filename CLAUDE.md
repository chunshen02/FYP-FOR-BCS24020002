# CLAUDE.md — Project Rules for AI Assistants

## Language

- All code, comments, variable names, function names, class names, docstrings,
  and commit messages must be written in **English**.
- No non-English identifiers or inline comments anywhere in the codebase.

## Verify Code Runs Before Finishing

Before reporting any task as complete:

1. **Syntax check** every modified file:
   ```
   python -m py_compile <file.py>
   ```
2. **Import check** for runtime entry points:
   ```
   python -c "import main"
   python -c "from runtime.detection.inference import DeepfakeInferenceEngine"
   python -c "from runtime.detection.preprocessor import Preprocessor"
   python -c "from runtime.detection.face_detector import FaceDetector"
   python -c "from runtime.threads.detection_worker import DetectionWorker"
   python -c "from runtime.storage.database import log_event"
   ```
3. If any check fails, fix the error before finishing — do not leave broken code.

## Project Stack & Pinned Versions

| Component        | Library / Tool        | Version       |
|------------------|-----------------------|---------------|
| Language         | Python                | 3.13.1        |
| GUI framework    | PyQt6                 | 6.8.1         |
| Screen capture   | mss                   | 9.0.2         |
| Face detection   | mediapipe             | 0.10.35       |
| Image processing | opencv-python         | 4.11.0.86     |
| HTTP client      | requests              | 2.32.3        |
| Env vars         | python-dotenv         | 1.1.0         |
| Numerics         | numpy                 | 2.2.5         |
| System metrics   | psutil                | (unpinned)    |
| Notifications    | plyer                 | (unpinned)    |
| Detection API    | BitMind Oracle API    | v1            |
| Local storage    | SQLite (stdlib)       | built-in      |

Do not upgrade pinned versions without testing. Do not add new dependencies
without updating `requirements.txt`.

## Architecture Overview

```
main.py                          # Entry point — launches PyQt6 app
config.py                        # All tuneable constants (thresholds, paths, timeouts)
runtime/
  capture/screen_capture.py      # mss screen grab → BGR numpy frame
  detection/
    face_detector.py             # MediaPipe BlazeFace → aligned face crops
    preprocessor.py              # BGR crop → JPEG bytes (cv2.imencode)
    inference.py                 # JPEG bytes → BitMind API → FaceResult + EMA
  threads/
    detection_worker.py          # QThread: capture → detect → infer → emit
    alert_thread.py              # QThread: watches FaceResult stream, fires alerts
  gui/
    main_window.py               # Main window, wires all components
    video_widget.py              # Live frame display widget
    session_log.py               # In-session event log panel
    dialogs.py                   # Save/discard session dialog
    alert_system.py              # Alert sound / notification helper
  storage/
    database.py                  # SQLite read/write for session event log
benchmark.py                     # Standalone pipeline performance benchmark
```

## Key Configuration (config.py)

- `BITMIND_API_KEY` — loaded from `.env` file (never hardcode)
- `FAKE_THRESHOLD = 0.70` — P_fake ≥ 0.70 → RED alert
- `WARNING_THRESHOLD = 0.50` — P_fake ≥ 0.50 → AMBER warning
- `EMA_ALPHA = 0.3` — smoothing factor for display score
- `ALERT_TRIGGER_FRAMES = 2` — consecutive RED frames to trigger alert
- `ALERT_DISMISS_FRAMES = 5` — consecutive non-RED frames to auto-dismiss

## API Setup

BitMind API key must be placed in a `.env` file at the project root:

```
BITMIND_API_KEY=your_key_here
```

The `.env` file is git-ignored. Never commit it.

## Coding Conventions

- No comments explaining *what* the code does — only *why* when non-obvious.
- No multi-line docstrings; one-line docstrings are acceptable.
- No dead code, unused imports, or backwards-compatibility shims.
- Prefer editing existing files over creating new ones.
- Do not add error handling for scenarios that cannot happen.
