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

## Do Not Run Scripts After Writing Them

- After writing or modifying a script (tests, evaluation, benchmark, data
  preparation, etc.), do **not** run it immediately. The user runs scripts
  themselves in the VS Code terminal so the output is visible to them.
- The syntax and import checks above are still required; they do not execute
  the script's main logic.
- Instead of running it, tell the user how to run it (the VS Code task name
  or the exact command) and what output to expect.

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
    inference.py                 # JPEG bytes → BitMind API → FaceResult
  threads/
    detection_worker.py          # QThread: capture → detect → infer → emit
    alert_thread.py              # QThread: watches FaceResult stream, fires alerts
  gui/
    main_window.py               # Main window, wires all components
    video_widget.py              # Live frame display widget
    session_log.py               # In-session event log panel
    dialogs.py                   # Save/discard session dialog
    alert_system.py              # Taskbar flash / beep / desktop notification on alert
  storage/
    database.py                  # SQLite read/write for session event log
```

## Key Configuration (config.py)

- `BITMIND_API_KEY` — loaded from `.env` file (never hardcode)
- `FAKE_THRESHOLD = 0.70` — P_fake ≥ 0.70 → DEEPFAKE; below → REAL (user-adjustable in Settings)
- `CONFIRM_VERDICTS = 3`, `CONFIRM_FAKE_VOTES = 2` — a first DEEPFAKE verdict is re-checked; 2 of 3 decide
- `MAX_PENDING_FACES = 3` — faces awaiting a BitMind verdict at once; all faces are tracked and drawn
- `DETECTION_TILE_GRID = 2` — N x N overlapping tiles for small-face detection (1 disables)
- `ALERT_TRIGGER_FRAMES = 3` — consecutive DEEPFAKE frames to trigger alert
- `ALERT_DISMISS_FRAMES = 8` — consecutive non-DEEPFAKE frames to auto-dismiss

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
