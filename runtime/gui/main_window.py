"""
main_window.py

Main application window matching the redesigned wireframe.
Wires all GUI components and background threads.
"""

import threading
from datetime import timedelta, datetime
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QStatusBar,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QProgressBar,
    QHeaderView,
    QAbstractItemView,
    QFrame,
    QDialog,
    QApplication,
)
from PyQt6.QtCore import pyqtSlot, Qt, QTimer
from PyQt6.QtGui import QColor, QFont
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from runtime.gui.video_widget import VideoWidget
from runtime.gui.session_log import SessionLog
from runtime.gui.dialogs import SaveSessionDialog
from runtime.threads.detection_worker import DetectionWorker
from runtime.threads.alert_thread import AlertThread
from runtime.detection.face_detector import FaceDetector
from runtime.detection.inference import DeepfakeInferenceEngine
from runtime.storage.database import log_event as db_log_event, get_all_sessions, delete_session, clear_history
import config


class MainWindow(QMainWindow):
    """
    Main application window for the real-time deepfake detector.
    Matches the reference wireframe exactly.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Real-Time Deepfake Detection System")
        self.setMinimumSize(900, 680)
        self.setStyleSheet(self._get_stylesheet())

        # State
        self._detection_worker: DetectionWorker = None
        self._alert_thread: AlertThread = None
        self._session_active = False
        self._session_start_time = None
        self._frame_count = 0
        self._current_alert_active = False
        self._alert_count = 0
        self._status_state = "Idle"

        # Shared heavy components — initialised once in background, reused across sessions
        self._shared_detector: FaceDetector = None
        self._shared_engine: DeepfakeInferenceEngine = None

        # Session timer
        self._session_timer = QTimer()
        self._session_timer.timeout.connect(self._tick_session_duration)

        self._build_ui()
        self.refresh_history()

        # Pre-init MediaPipe and HTTP session while the user is reading the UI
        threading.Thread(target=self._preinit_components, daemon=True).start()

    # -- UI Construction -------------------------------------------------------

    def _build_ui(self):
        """Build the main window layout according to wireframe."""
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        
        # Main layout: vertical with no margins/spacing
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # --- TOP SECTION: LEFT PANEL (video) + RIGHT PANEL (cards) ---
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # LEFT PANEL: Video Feed + Controls
        left_panel = self._build_left_panel()
        self._video_widget.region_selected.connect(self._on_region_selected)
        top_splitter.addWidget(left_panel)
        top_splitter.setStretchFactor(0, 60)

        # RIGHT PANEL: Info Cards
        right_panel = self._build_right_panel()
        top_splitter.addWidget(right_panel)
        top_splitter.setStretchFactor(1, 40)

        root_layout.addWidget(top_splitter, stretch=1)

        # --- MIDDLE SECTION: Session History Table (wrapped in container) ---
        self._history_section = QWidget()
        history_layout = QVBoxLayout(self._history_section)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(0)

        # Header row: title + delete buttons
        history_header = QHBoxLayout()
        history_header.setContentsMargins(8, 8, 8, 4)
        history_header.setSpacing(8)
        history_title = QLabel("Session History")
        history_title.setStyleSheet("color: #888; font-size: 11px;")
        history_header.addWidget(history_title)
        history_header.addStretch()
        self._btn_delete_session = QPushButton("Delete Selected")
        self._btn_delete_session.setObjectName("btn_delete_session")
        self._btn_delete_session.setFixedHeight(24)
        self._btn_delete_session.setEnabled(False)
        self._btn_delete_session.clicked.connect(self._on_delete_selected_session)
        history_header.addWidget(self._btn_delete_session)
        self._btn_clear_history = QPushButton("Clear All")
        self._btn_clear_history.setObjectName("btn_clear_history")
        self._btn_clear_history.setFixedHeight(24)
        self._btn_clear_history.clicked.connect(self._on_clear_all_sessions)
        history_header.addWidget(self._btn_clear_history)
        history_layout.addLayout(history_header)

        self._history_table = self._build_history_table()
        self._history_table.itemSelectionChanged.connect(self._on_history_selection_changed)
        history_layout.addWidget(self._history_table)
        root_layout.addWidget(self._history_section, stretch=0)

        # --- BOTTOM SECTION: Status Bar ---
        self.setStatusBar(QStatusBar())
        self._update_status_bar()

    def _build_left_panel(self) -> QWidget:
        """Build left panel: video feed + control bar."""
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # Video widget
        self._video_widget = VideoWidget()
        self._video_widget.setMinimumHeight(300)
        left_layout.addWidget(self._video_widget, stretch=1)

        # Control bar
        control_bar = self._build_control_bar()
        left_layout.addLayout(control_bar)

        return left_container

    def _build_control_bar(self) -> QHBoxLayout:
        """Build control bar with buttons."""
        control_bar = QHBoxLayout()
        control_bar.setContentsMargins(12, 10, 12, 10)
        control_bar.setSpacing(8)

        # Style: add top border via stylesheet trick
        # (will be applied to buttons)

        # Start button
        self._btn_start = QPushButton("▶ Start")
        self._btn_start.setObjectName("btn_start")
        self._btn_start.setFixedHeight(32)
        self._btn_start.clicked.connect(self._on_start)
        control_bar.addWidget(self._btn_start)

        # Stop button
        self._btn_stop = QPushButton("◻ Stop")
        self._btn_stop.setObjectName("btn_stop")
        self._btn_stop.setFixedHeight(32)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        control_bar.addWidget(self._btn_stop)

        # Screenshot button
        self._btn_screenshot = QPushButton("📷 Screenshot")
        self._btn_screenshot.setObjectName("btn_screenshot")
        self._btn_screenshot.setFixedHeight(32)
        self._btn_screenshot.clicked.connect(self._on_screenshot)
        control_bar.addWidget(self._btn_screenshot)

        # Clear Zone button — enabled only when a detection zone is active
        self._btn_clear_zone = QPushButton("✕ Clear Zone")
        self._btn_clear_zone.setObjectName("btn_clear_zone")
        self._btn_clear_zone.setFixedHeight(32)
        self._btn_clear_zone.setEnabled(False)
        self._btn_clear_zone.setToolTip("Drag on the video to set a detection zone. Right-click to clear.")
        self._btn_clear_zone.clicked.connect(self._on_clear_zone)
        control_bar.addWidget(self._btn_clear_zone)

        control_bar.addStretch()

        # FPS readout
        fps_container = QHBoxLayout()
        fps_container.setSpacing(4)
        fps_icon = QLabel("FPS")
        fps_icon.setStyleSheet("font-size: 11px; color: #666666;")
        self._fps_label = QLabel("—")
        self._fps_label.setStyleSheet(
            "font-size: 12px; font-weight: 500; color: #aaaaaa; min-width: 28px;"
        )
        fps_container.addWidget(fps_icon)
        fps_container.addWidget(self._fps_label)

        # Vertical separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #333333;")

        # Faces readout
        faces_container = QHBoxLayout()
        faces_container.setSpacing(4)
        faces_icon = QLabel("Faces")
        faces_icon.setStyleSheet("font-size: 11px; color: #666666;")
        self._faces_label = QLabel("—")
        self._faces_label.setStyleSheet(
            "font-size: 12px; font-weight: 500; color: #aaaaaa; min-width: 20px;"
        )
        faces_container.addWidget(faces_icon)
        faces_container.addWidget(self._faces_label)

        # Another separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.VLine)
        sep2.setStyleSheet("color: #333333;")

        # API response time readout
        self._lbl_api_ms = QLabel("API: — ms")
        self._lbl_api_ms.setStyleSheet("font-weight: bold; padding: 4px 12px;")

        control_bar.addLayout(fps_container)
        control_bar.addWidget(sep)
        control_bar.addLayout(faces_container)
        control_bar.addWidget(sep2)
        control_bar.addWidget(self._lbl_api_ms)

        return control_bar

    def _build_right_panel(self) -> QWidget:
        """Build right panel with info cards."""
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(12, 0, 0, 0)
        right_layout.setSpacing(10)

        # Confidence Card
        self._confidence_card = self._build_confidence_card()
        right_layout.addWidget(self._confidence_card)

        # Alert Card (hidden by default)
        self._alert_card = self._build_alert_card()
        self._alert_card.setVisible(False)
        right_layout.addWidget(self._alert_card)

        # Session Info Card
        self._session_info_card = self._build_session_info_card()
        right_layout.addWidget(self._session_info_card)

        # Session Log
        self._session_log = SessionLog()
        self._session_log.setObjectName("session_log")
        right_layout.addWidget(self._session_log, stretch=1)

        return right_container

    def _build_confidence_card(self) -> QWidget:
        """Build confidence score card."""
        card = QWidget()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(8)

        # Title
        title = QLabel("Confidence score")
        title.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(title)

        # Progress bar + percentage row
        progress_row = QHBoxLayout()
        progress_row.setSpacing(8)

        self._confidence_bar = QProgressBar()
        self._confidence_bar.setObjectName("confidence_bar")
        self._confidence_bar.setRange(0, 100)
        self._confidence_bar.setValue(0)
        self._confidence_bar.setFixedHeight(7)
        self._confidence_bar.setTextVisible(False)
        progress_row.addWidget(self._confidence_bar, stretch=1)

        self._confidence_label = QLabel("—")
        self._confidence_label.setStyleSheet(
            "color: #EF9F27; font-size: 15px; font-weight: bold;"
        )
        self._confidence_label.setFixedWidth(60)
        progress_row.addWidget(self._confidence_label)

        layout.addLayout(progress_row)
        return card

    def _build_alert_card(self) -> QWidget:
        """Build alert card (shown when alert triggered)."""
        card = QWidget()
        card.setObjectName("alert_card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(10)

        # Header
        header_layout = QHBoxLayout()
        icon_label = QLabel("⚠")
        icon_label.setStyleSheet("color: #633806; font-size: 14px;")
        header_label = QLabel("Deepfake detected")
        header_label.setStyleSheet(
            "color: #633806; font-size: 12px; font-weight: bold;"
        )
        header_layout.addWidget(icon_label)
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Acknowledge button
        self._btn_acknowledge = QPushButton("Acknowledge alert")
        self._btn_acknowledge.setObjectName("btn_acknowledge")
        self._btn_acknowledge.clicked.connect(self._on_acknowledge_alert)
        layout.addWidget(self._btn_acknowledge)

        return card

    def _build_session_info_card(self) -> QWidget:
        """Build session info card."""
        card = QWidget()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(0)

        # Title
        title = QLabel("Session info")
        title.setStyleSheet("color: #888; font-size: 11px; margin-bottom: 8px;")
        layout.addWidget(title)

        # Status row
        status_row_layout = QHBoxLayout()
        status_row_layout.setContentsMargins(0, 6, 0, 6)
        status_row_layout.setSpacing(12)
        status_label = QLabel("Status")
        status_label.setStyleSheet("color: #888; font-size: 11px;")
        status_row_layout.addWidget(status_label)
        status_row_layout.addStretch()
        self._status_value = QLabel("● Active")
        self._status_value.setStyleSheet("color: #ddd; font-size: 11px;")
        status_row_layout.addWidget(self._status_value)
        layout.addLayout(status_row_layout)

        # Duration row
        duration_row_layout = QHBoxLayout()
        duration_row_layout.setContentsMargins(0, 6, 0, 6)
        duration_row_layout.setSpacing(12)
        duration_label = QLabel("Duration")
        duration_label.setStyleSheet("color: #888; font-size: 11px;")
        duration_row_layout.addWidget(duration_label)
        duration_row_layout.addStretch()
        self._duration_value = QLabel("00:00:00")
        self._duration_value.setStyleSheet("color: #ddd; font-size: 11px;")
        duration_row_layout.addWidget(self._duration_value)
        layout.addLayout(duration_row_layout)

        # Frames row
        frames_row_layout = QHBoxLayout()
        frames_row_layout.setContentsMargins(0, 6, 0, 6)
        frames_row_layout.setSpacing(12)
        frames_label = QLabel("Frames")
        frames_label.setStyleSheet("color: #888; font-size: 11px;")
        frames_row_layout.addWidget(frames_label)
        frames_row_layout.addStretch()
        self._frames_value = QLabel("0")
        self._frames_value.setStyleSheet("color: #ddd; font-size: 11px;")
        frames_row_layout.addWidget(self._frames_value)
        layout.addLayout(frames_row_layout)

        # Model row
        model_row_layout = QHBoxLayout()
        model_row_layout.setContentsMargins(0, 6, 0, 6)
        model_row_layout.setSpacing(12)
        model_label = QLabel("Model")
        model_label.setStyleSheet("color: #888; font-size: 11px;")
        model_row_layout.addWidget(model_label)
        model_row_layout.addStretch()
        model_value = QLabel(config.MODEL_NAME.replace("_", "-").title())
        model_value.setStyleSheet("color: #ddd; font-size: 11px;")
        model_row_layout.addWidget(model_value)
        layout.addLayout(model_row_layout)

        return card

    def _build_history_table(self) -> QTableWidget:
        """Build session history table."""
        table = QTableWidget()
        table.setObjectName("history_table")
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(
            ["Date & time", "Duration", "Alerts", "Result"]
        )
        table.setShowGrid(False)
        table.setAlternatingRowColors(False)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(False)
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Fixed
        )
        table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Fixed
        )
        table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Fixed
        )
        table.setColumnWidth(1, 100)
        table.setColumnWidth(2, 70)
        table.setColumnWidth(3, 160)
        table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        table.setSelectionMode(
            QTableWidget.SelectionMode.SingleSelection
        )
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        table.setFixedHeight(150)

        return table

    def _update_status_bar(self):
        """Update status bar with current state."""
        status_bar = self.statusBar()
        status_bar.clearMessage()
        status_bar.setStyleSheet(
            "background-color: #222222; color: #888888; font-size: 11px;"
        )

        # Create status labels
        lbl_source = QLabel("Source: Zoom")
        lbl_source.setStyleSheet("color: #888;")

        lbl_model = QLabel("● Model loaded")
        lbl_model.setStyleSheet("color: #888;")

        lbl_db = QLabel("● DB connected")
        lbl_db.setStyleSheet("color: #888;")

        lbl_threshold = QLabel(f"Threshold: {config.FAKE_THRESHOLD * 100:.0f}%")
        lbl_threshold.setStyleSheet("color: #888;")

        status_bar.addPermanentWidget(lbl_source)
        status_bar.addPermanentWidget(QLabel(" · "))
        status_bar.addPermanentWidget(lbl_model)
        status_bar.addPermanentWidget(QLabel(" · "))
        status_bar.addPermanentWidget(lbl_db)
        status_bar.addPermanentWidget(QLabel(" · "))
        status_bar.addPermanentWidget(lbl_threshold)

    def _preinit_components(self):
        """Load MediaPipe model and open HTTP session once at startup."""
        try:
            detector = FaceDetector()
            engine = DeepfakeInferenceEngine()
            self._shared_detector = detector
            self._shared_engine = engine
        except Exception:
            pass  # Worker falls back to creating its own and surfaces any API key error

    def refresh_history(self):
        """Reload session history from database and update table visibility."""
        sessions = get_all_sessions(limit=100)
        if not sessions:
            self._history_section.setVisible(False)
        else:
            self._history_section.setVisible(True)
            self.populate_history_table(sessions)

    def populate_history_table(self, sessions: list):
        """Populate history table with session data."""
        self._history_table.setRowCount(0)  # clear first

        for row_data in sessions:
            row = self._history_table.rowCount()
            self._history_table.insertRow(row)

            # Col 0 — Date & time
            dt_item = QTableWidgetItem(row_data["started_at"])
            dt_item.setTextAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            dt_item.setForeground(QColor("#ddd"))
            self._history_table.setItem(row, 0, dt_item)

            # Col 1 — Duration
            dur_item = QTableWidgetItem(row_data["duration"])
            dur_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            dur_item.setForeground(QColor("#ddd"))
            self._history_table.setItem(row, 1, dur_item)

            # Col 2 — Alerts
            alert_item = QTableWidgetItem(str(row_data["alert_count"]))
            alert_item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter
            )
            alert_item.setForeground(QColor("#ddd"))
            self._history_table.setItem(row, 2, alert_item)

            # Col 3 — Result badge
            result_text = row_data["result"]
            badge = QLabel(result_text)
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            if result_text == "Deepfake detected":
                badge.setStyleSheet(
                    """
                    background-color: #FCEBEB;
                    color: #A32D2D;
                    border-radius: 3px;
                    padding: 2px 8px;
                    font-size: 11px;
                    font-weight: 500;
                """
                )
            else:
                badge.setStyleSheet(
                    """
                    background-color: #EAF3DE;
                    color: #3B6D11;
                    border-radius: 3px;
                    padding: 2px 8px;
                    font-size: 11px;
                    font-weight: 500;
                """
                )
            # Wrap badge in a centered container
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.addWidget(badge)
            layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.setContentsMargins(4, 2, 4, 2)
            self._history_table.setCellWidget(row, 3, container)

    @pyqtSlot()
    def _on_history_selection_changed(self):
        self._btn_delete_session.setEnabled(self._history_table.currentRow() >= 0)

    @pyqtSlot()
    def _on_delete_selected_session(self):
        row = self._history_table.currentRow()
        if row < 0:
            return
        started_at = self._history_table.item(row, 0).text()
        delete_session(started_at)
        self.refresh_history()

    @pyqtSlot()
    def _on_clear_all_sessions(self):
        clear_history()
        self.refresh_history()

    # -- Detection Control -----------------------------------------------------

    def _set_status(self, state: str):
        """
        Update the Status row dot + label in the Session info card.

        state: one of "Idle", "Initializing", "Detecting", "Alert", "Stopping".
        """
        configs = {
            "Idle":         ("#666666", "#888888", "Idle"),
            "Initializing": ("#EF9F27", "#633806", "Initializing…"),
            "Detecting":    ("#1D9E75", "#085041", "Active"),
            "Alert":        ("#E24B4A", "#791F1F", "Alert"),
            "Stopping":     ("#EF9F27", "#633806", "Stopping…"),
        }
        dot_color, text_color, label = configs.get(state, configs["Idle"])
        self._status_state = state
        self._status_value.setText(
            f"<span style='color:{dot_color};'>●</span> "
            f"<span style='color:{text_color};'>{label}</span>"
        )
        self._status_value.setStyleSheet("font-size: 11px;")

    @pyqtSlot()
    def _on_start(self):
        """Start detection and alert threads."""
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._session_active = True
        self._session_start_time = 0
        self._frame_count = 0
        self._current_alert_active = False
        self._alert_count = 0

        # Hide alert card
        self._alert_card.setVisible(False)

        # Reset confidence
        self._confidence_bar.setValue(0)
        self._confidence_label.setText("—")

        # Reset stats readouts
        self._fps_label.setText("—")
        self._faces_label.setText("—")

        # Update session info
        self._set_status("Initializing")
        self._duration_value.setText("00:00:00")
        self._frames_value.setText("0")

        # Start session timer
        self._session_timer.start(1000)

        # Detection worker — reuse pre-initialised components to avoid per-session reload
        self._detection_worker = DetectionWorker(
            face_detector=self._shared_detector,
            engine=self._shared_engine,
        )
        self._detection_worker.set_exclusion_region(self._window_screen_rect())
        if self._video_widget.selection_rect is not None:
            self._detection_worker.set_focus_region(self._video_widget.selection_rect)
        self._detection_worker.frame_ready.connect(
            self._video_widget.update_frame
        )
        self._detection_worker.frame_ready.connect(self._on_frame_ready)
        self._detection_worker.fps_update.connect(self._on_fps_update)
        self._detection_worker.api_response_update.connect(self._on_api_response_update)
        self._detection_worker.error_occurred.connect(self._on_error)

        # Alert thread
        self._alert_thread = AlertThread()
        self._alert_thread.alert_triggered.connect(self._on_alert_triggered)
        self._alert_thread.alert_dismissed.connect(self._on_alert_dismissed)

        # Wire detection results to alert thread
        self._detection_worker.frame_ready.connect(
            lambda frame, results: self._alert_thread.push_results(results)
        )

        self._alert_thread.start()
        self._detection_worker.start()

        self._session_log_event("SESSION", "Detection started.")
        db_log_event("SESSION", "Detection started.")

    @pyqtSlot()
    def _on_stop(self):
        """Stop detection and alert threads, then prompt to save or discard."""
        # Show "Stopping…" immediately and force a repaint before any blocking work
        self._set_status("Stopping")
        QApplication.processEvents()

        self._btn_stop.setEnabled(False)
        was_active = self._session_active
        self._session_active = False
        self._session_timer.stop()

        if self._detection_worker:
            self._detection_worker.stop()
            self._detection_worker.wait(3000)
        if self._alert_thread:
            self._alert_thread.stop()
            self._alert_thread.wait(2000)

        self._btn_start.setEnabled(True)
        self._alert_card.setVisible(False)
        self._video_widget.clear()
        self._lbl_api_ms.setText("API: — ms")

        # Prompt the user only if a session was actually running
        if was_active:
            duration_str = self._duration_value.text()
            alert_count = self._alert_count

            dialog = SaveSessionDialog(
                duration=duration_str,
                alert_count=alert_count,
                parent=self,
            )
            result = dialog.exec()

            if result == QDialog.DialogCode.Accepted:
                # Save: log the stop event so the session is persisted in history
                self._session_log_event("SESSION", "Detection stopped.")
                db_log_event("SESSION", "Detection stopped.")
                self.refresh_history()
            # Discard: skip writing the stop event so the session is not surfaced

        # Return UI to Idle state and reset stats readouts
        self._set_status("Idle")
        self._fps_label.setText("—")
        self._faces_label.setText("—")

    @pyqtSlot()
    def _on_screenshot(self):
        """Save the current video frame as a PNG to the screenshots folder."""
        pixmap = self._video_widget.grab_screenshot()
        if pixmap is None:
            self._session_log_event("INFO", "Screenshot skipped: no active feed.")
            return
        screenshots_dir = os.path.join(config.BASE_DIR, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        path = os.path.join(screenshots_dir, filename)
        pixmap.save(path)
        self._session_log_event("INFO", f"Screenshot saved: {filename}")

    @pyqtSlot()
    def _on_acknowledge_alert(self):
        """Acknowledge the current alert."""
        self._alert_card.setVisible(False)
        self._current_alert_active = False
        if self._session_active:
            self._set_status("Detecting")
        self._session_log_event("DISMISS", "Alert acknowledged.")
        db_log_event("DISMISS", "Alert acknowledged.")

    # -- Frame & Result Handling -----------------------------------------------

    @pyqtSlot(object, object)
    def _on_frame_ready(self, frame: np.ndarray, results: list):
        """Handle incoming frame and detection results."""
        self._frame_count += 1
        self._frames_value.setText(f"{self._frame_count:,}")

        # First frame after start transitions Initializing -> Detecting
        if self._status_state == "Initializing":
            self._set_status("Detecting")

        # Update face count readout
        self._faces_label.setText(str(len(results) if results else 0))

        # Update confidence based on worst result
        if results:
            worst = max(results, key=lambda r: r.p_fake_raw)
            # Use raw score for accuracy — real-time API response
            display_confidence = worst.p_fake_raw * 100
            self._confidence_bar.setValue(int(display_confidence))
            self._confidence_label.setText(f"{display_confidence:.1f}%")

            if worst.p_fake_raw >= config.FAKE_THRESHOLD and not self._current_alert_active:
                self._show_alert()
        else:
            self._confidence_bar.setValue(0)
            self._confidence_label.setText("—")

    @pyqtSlot(float)
    def _on_fps_update(self, fps: float):
        """Update the FPS readout in the control bar."""
        self._fps_label.setText(str(round(fps)))

    @pyqtSlot(float)
    def _on_api_response_update(self, api_ms: float):
        self._lbl_api_ms.setText(f"API: {api_ms:.0f} ms")

    @pyqtSlot(float)
    def _on_alert_triggered(self, peak_p_fake: float):
        """Handle alert trigger from alert thread."""
        if not self._current_alert_active:
            self._show_alert()
        self._session_log_event(
            "ALERT",
            f"Deepfake detected. Peak confidence: {peak_p_fake:.1%}"
        )
        db_log_event(
            "ALERT",
            f"Deepfake detected. Peak confidence: {peak_p_fake:.1%}",
            peak_confidence=peak_p_fake
        )

    @pyqtSlot()
    def _on_alert_dismissed(self):
        """Handle alert dismissal from alert thread."""
        self._alert_card.setVisible(False)
        self._current_alert_active = False
        if self._session_active:
            self._set_status("Detecting")
        self._session_log_event("DISMISS", "Alert dismissed automatically.")
        db_log_event("DISMISS", "Alert dismissed automatically.")

    @pyqtSlot(str)
    def _on_error(self, error_msg: str):
        """Handle errors from detection worker."""
        self._session_log_event("ERROR", error_msg)
        db_log_event("ERROR", error_msg)
        self._on_stop()

    def _show_alert(self):
        """Show alert card and reflect the Alert state."""
        if not self._current_alert_active:
            self._alert_count += 1
        self._alert_card.setVisible(True)
        self._current_alert_active = True
        self._set_status("Alert")

    def _tick_session_duration(self):
        """Update session duration every second."""
        if self._session_active:
            self._session_start_time += 1
            duration_str = str(timedelta(seconds=self._session_start_time))
            self._duration_value.setText(duration_str)

    def _session_log_event(self, event_type: str, detail: str):
        """Append a timestamped entry to the session log widget."""
        self._session_log.log_event(event_type, detail)

    # -- Stylesheet -----------------------------------------------------------

    @staticmethod
    def _get_stylesheet() -> str:
        """Return the complete stylesheet for the application."""
        return """
QMainWindow, QWidget#central {
    background-color: #1a1a1a;
    color: #dddddd;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}

/* Control Bar */
QHBoxLayout {
    border-top: 1px solid #3a3a3a;
}

/* Buttons */
QPushButton#btn_start {
    background-color: #E1F5EE;
    border: 1px solid #1D9E75;
    color: #085041;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: bold;
}

QPushButton#btn_start:hover {
    background-color: #C8EDE8;
}

QPushButton#btn_start:pressed {
    background-color: #B0E5E0;
}

QPushButton#btn_stop {
    background-color: #FCEBEB;
    border: 1px solid #E24B4A;
    color: #791F1F;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: bold;
}

QPushButton#btn_stop:hover {
    background-color: #F5D3D3;
}

QPushButton#btn_stop:pressed {
    background-color: #EEBBBB;
}

QPushButton#btn_screenshot {
    background-color: #2d2d2d;
    border: 1px solid #555555;
    color: #dddddd;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
}

QPushButton#btn_screenshot:hover {
    background-color: #3a3a3a;
    border-color: #777777;
}

QPushButton#btn_screenshot:pressed {
    background-color: #444444;
}

QPushButton#btn_clear_zone {
    background-color: #2d2d2d;
    border: 1px solid #4A9FEF;
    color: #4A9FEF;
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 12px;
}

QPushButton#btn_clear_zone:disabled {
    background-color: #2d2d2d;
    border: 1px solid #444444;
    color: #555555;
}

QPushButton#btn_clear_zone:hover {
    background-color: #3a3a3a;
}

QPushButton#btn_acknowledge {
    background-color: #FAC775;
    border: 1px solid #BA7517;
    color: #412402;
    border-radius: 5px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: bold;
}

QPushButton#btn_acknowledge:hover {
    background-color: #F2B860;
}

/* Cards */
QWidget#card, QWidget#alert_card {
    background-color: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
}

QWidget#alert_card {
    background-color: #FAEEDA;
    border: 1px solid #EF9F27;
}

/* Progress Bar */
QProgressBar#confidence_bar {
    border: none;
    background-color: #444444;
    border-radius: 3px;
    height: 7px;
}

QProgressBar#confidence_bar::chunk {
    background-color: #EF9F27;
    border-radius: 3px;
}

/* Table */
QTableWidget#history_table {
    background-color: #1a1a1a;
    color: #cccccc;
    border: none;
    gridline-color: #2a2a2a;
    alternate-background-color: #1a1a1a;
}

QTableWidget#history_table::item {
    padding: 6px 8px;
    border-bottom: 1px solid #2a2a2a;
}

QHeaderView::section {
    background-color: transparent;
    color: #888888;
    font-size: 11px;
    border: none;
    border-bottom: 1px solid #333333;
    padding: 4px 8px;
}

QPushButton#btn_delete_session {
    background-color: #2d2d2d;
    border: 1px solid #E24B4A;
    color: #E24B4A;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 11px;
}
QPushButton#btn_delete_session:disabled {
    border-color: #444444;
    color: #555555;
}
QPushButton#btn_delete_session:hover {
    background-color: #3a2020;
}

QPushButton#btn_clear_history {
    background-color: #2d2d2d;
    border: 1px solid #555555;
    color: #999999;
    border-radius: 4px;
    padding: 2px 10px;
    font-size: 11px;
}
QPushButton#btn_clear_history:hover {
    background-color: #3a2020;
    border-color: #E24B4A;
    color: #E24B4A;
}

/* Status Bar */
QStatusBar {
    background-color: #222222;
    color: #888888;
    font-size: 11px;
    border-top: 1px solid #333333;
}

/* Video Widget */
QWidget#video_widget {
    background-color: #000000;
}

/* Session Log */
QTextEdit#session_log {
    background-color: #111111;
    color: #cccccc;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    font-family: 'Consolas', monospace;
    font-size: 11px;
    padding: 4px;
}
"""

    # -- Detection zone -------------------------------------------------------

    @pyqtSlot(object)
    def _on_region_selected(self, rect):
        if self._detection_worker is not None:
            self._detection_worker.set_focus_region(rect)
        self._btn_clear_zone.setEnabled(rect is not None)

    @pyqtSlot()
    def _on_clear_zone(self):
        self._video_widget.clear_selection()

    # -- Window geometry exclusion --------------------------------------------

    def _window_screen_rect(self) -> tuple:
        """Return window frame bounds in physical pixels for mss exclusion."""
        ratio = self.devicePixelRatioF()
        geo = self.frameGeometry()
        return (
            int(geo.x() * ratio),
            int(geo.y() * ratio),
            int(geo.width() * ratio),
            int(geo.height() * ratio),
        )

    def _sync_exclusion(self):
        if self._detection_worker and self._session_active:
            self._detection_worker.set_exclusion_region(self._window_screen_rect())

    def moveEvent(self, event):
        super().moveEvent(event)
        self._sync_exclusion()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_exclusion()

    # -- Cleanup ---------------------------------------------------------------

    def closeEvent(self, event):
        """Ensure threads are stopped cleanly on window close."""
        self._session_timer.stop()
        self._on_stop()
        if self._shared_detector:
            self._shared_detector.close()
        if self._shared_engine:
            self._shared_engine.close()
        event.accept()
