"""
main_window.py

Main application window matching the redesigned wireframe.
Wires all GUI components and background threads.
"""

import threading
import time
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
    QHeaderView,
    QAbstractItemView,
    QFrame,
    QDialog,
    QApplication,
    QScrollArea,
)
from PyQt6.QtCore import pyqtSlot, Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QColor, QFont, QIcon, QPixmap, QPainter, QPainterPath, QPen, QBrush
import numpy as np
import cv2
import mss
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from runtime.gui.video_widget import VideoWidget
from runtime.gui.session_log import SessionLog
from runtime.gui.dialogs import SaveSessionDialog
from runtime.gui.gauge_widget import GaugeWidget
from runtime.gui.alert_system import notify_alert
from runtime.capture.window_affinity import (
    apply_capture_exclusion,
    clear_capture_exclusion,
    restore_capture_exclusion,
    capture_affinity_lock,
)
from runtime.gui.face_thumbnail import FaceThumbnail
from runtime.threads.detection_worker import DetectionWorker
from runtime.threads.alert_thread import AlertThread
from runtime.detection.face_detector import FaceDetector
from runtime.detection.inference import DeepfakeInferenceEngine
from runtime.storage.database import (
    log_event as db_log_event, get_all_sessions, delete_session, delete_events_from, clear_history,
)
import config


def _build_shield_icon(colour: str = "#22C55E", size: int = 64) -> QIcon:
    """Paint a simple shield glyph for the window/taskbar icon (no asset file needed)."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    path = QPainterPath()
    m = size * 0.12
    path.moveTo(size / 2, m * 0.4)
    path.lineTo(size - m, m * 1.4)
    path.lineTo(size - m, size * 0.55)
    path.cubicTo(size - m, size * 0.8, size * 0.75, size * 0.95, size / 2, size - m * 0.4)
    path.cubicTo(size * 0.25, size * 0.95, m, size * 0.8, m, size * 0.55)
    path.lineTo(m, m * 1.4)
    path.closeSubpath()

    painter.setPen(QPen(QColor(colour).darker(130), max(1, size // 24)))
    painter.setBrush(QBrush(QColor(colour)))
    painter.drawPath(path)

    # Checkmark
    check = QPainterPath()
    check.moveTo(size * 0.32, size * 0.5)
    check.lineTo(size * 0.45, size * 0.63)
    check.lineTo(size * 0.70, size * 0.35)
    painter.setPen(QPen(QColor("#0B0E14"), max(2, size // 12), cap=Qt.PenCapStyle.RoundCap, join=Qt.PenJoinStyle.RoundJoin))
    painter.drawPath(check)

    painter.end()
    return QIcon(pixmap)


class MainWindow(QMainWindow):
    """
    Main application window for the real-time deepfake detector.
    Matches the reference wireframe exactly.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Real-Time Deepfake Detection System")
        self.setWindowIcon(_build_shield_icon())
        self.setMinimumSize(820, 600)
        self.setStyleSheet(self._get_stylesheet())

        # State
        self._detection_worker: DetectionWorker = None
        self._alert_thread: AlertThread = None
        self._session_active = False
        self._session_start_time = None
        self._session_start_id = None   # DB id of this session's start event, so Discard can delete it
        self._frame_count = 0
        self._current_alert_active = False
        self._alert_count = 0
        self._status_state = "Idle"
        self._degraded = False
        self._capture_exclusion_applied = False
        self._capture_affinity_value = None   # WDA_* constant actually applied, for temporary clear/restore
        self._mini_mode = False
        self._full_mode_geometry = None
        # (face_id, last_updated) of what is currently rendered, so per-frame
        # updates skip the costly setStyleSheet/QPixmap work when nothing changed.
        self._analysis_key = None
        self._mini_strip_key = None

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
        
        # Main layout: padded so cards have breathing room from the window edge
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(14, 10, 14, 8)
        root_layout.setSpacing(10)

        # --- TOP SECTION: LEFT PANEL (video) + RIGHT PANEL (cards) ---
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setHandleWidth(10)

        # LEFT PANEL: Video Feed + Controls
        self._left_panel = self._build_left_panel()
        self._video_widget.region_selected.connect(self._on_region_selected)
        self._video_widget.recheck_requested.connect(self._on_recheck_requested)
        top_splitter.addWidget(self._left_panel)
        top_splitter.setStretchFactor(0, 55)

        # RIGHT PANEL: Info Cards
        right_panel = self._build_right_panel()
        top_splitter.addWidget(right_panel)
        top_splitter.setStretchFactor(1, 45)
        self._top_splitter = top_splitter

        root_layout.addWidget(top_splitter, stretch=1)

        # --- MIDDLE SECTION: Session History Table (wrapped in container) ---
        self._history_section = QWidget()
        history_layout = QVBoxLayout(self._history_section)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(0)

        # Header row: title + delete buttons
        history_header = QHBoxLayout()
        history_header.setContentsMargins(8, 6, 8, 3)
        history_header.setSpacing(6)
        history_title = QLabel("Session History")
        history_title.setStyleSheet("color: #888; font-size: 10px;")
        history_header.addWidget(history_title)
        history_header.addStretch()
        self._btn_delete_session = QPushButton("Delete Selected")
        self._btn_delete_session.setObjectName("btn_delete_session")
        self._btn_delete_session.setFixedHeight(20)
        self._btn_delete_session.setEnabled(False)
        self._btn_delete_session.clicked.connect(self._on_delete_selected_session)
        history_header.addWidget(self._btn_delete_session)
        self._btn_clear_history = QPushButton("Clear All")
        self._btn_clear_history.setObjectName("btn_clear_history")
        self._btn_clear_history.setFixedHeight(20)
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
        left_layout.setSpacing(6)

        # "LIVE PREVIEW" section header with a small green status dot
        preview_header = QHBoxLayout()
        preview_header.setSpacing(5)
        live_dot = QLabel("●")
        live_dot.setStyleSheet("color: #22C55E; font-size: 8px;")
        preview_header.addWidget(live_dot)
        preview_label = QLabel("LIVE PREVIEW")
        preview_label.setStyleSheet(
            "color: #6B7280; font-size: 10px; font-weight: 600; letter-spacing: 1px;"
        )
        preview_header.addWidget(preview_label)
        preview_header.addStretch()
        left_layout.addLayout(preview_header)

        # Video widget, framed with a rounded card border
        video_frame = QWidget()
        video_frame.setObjectName("video_frame")
        video_frame_layout = QVBoxLayout(video_frame)
        video_frame_layout.setContentsMargins(2, 2, 2, 2)
        self._video_widget = VideoWidget()
        self._video_widget.setMinimumHeight(260)
        video_frame_layout.addWidget(self._video_widget)
        left_layout.addWidget(video_frame, stretch=1)

        # Low-light warning bar (hidden until triggered)
        self._low_light_bar = QLabel("  Low light detected — detection accuracy may be reduced")
        self._low_light_bar.setStyleSheet(
            "background-color: #5C3500; color: #FFCC66; font-size: 10px; padding: 2px 6px; border-radius: 5px;"
        )
        self._low_light_bar.setVisible(False)
        left_layout.addWidget(self._low_light_bar)

        # Control bar
        control_bar = self._build_control_bar()
        left_layout.addLayout(control_bar)

        return left_container

    def _build_control_bar(self) -> QHBoxLayout:
        """Build control bar with buttons."""
        control_bar = QHBoxLayout()
        control_bar.setContentsMargins(10, 8, 10, 8)
        control_bar.setSpacing(6)

        # Style: add top border via stylesheet trick
        # (will be applied to buttons)

        # Start button
        self._btn_start = QPushButton("▶ Start")
        self._btn_start.setObjectName("btn_start")
        self._btn_start.setFixedHeight(28)
        self._btn_start.clicked.connect(self._on_start)
        control_bar.addWidget(self._btn_start)

        # Stop button
        self._btn_stop = QPushButton("◻ Stop")
        self._btn_stop.setObjectName("btn_stop")
        self._btn_stop.setFixedHeight(28)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        control_bar.addWidget(self._btn_stop)

        # Screenshot button
        self._btn_screenshot = QPushButton("📷 Screenshot")
        self._btn_screenshot.setObjectName("btn_screenshot")
        self._btn_screenshot.setFixedHeight(28)
        self._btn_screenshot.clicked.connect(self._on_screenshot)
        control_bar.addWidget(self._btn_screenshot)

        # Clear Zone button — enabled only when a detection zone is active
        self._btn_clear_zone = QPushButton("✕ Clear Zone")
        self._btn_clear_zone.setObjectName("btn_clear_zone")
        self._btn_clear_zone.setFixedHeight(28)
        self._btn_clear_zone.setEnabled(False)
        self._btn_clear_zone.setToolTip("Drag on the video to set a detection zone. Right-click to clear.")
        self._btn_clear_zone.clicked.connect(self._on_clear_zone)
        control_bar.addWidget(self._btn_clear_zone)

        # Mini mode — collapses to a small always-on-top status bar so the
        # meeting window underneath stays visible during a call.
        self._btn_mini_mode = QPushButton("⛶ Mini Mode")
        self._btn_mini_mode.setObjectName("btn_clear_zone")
        self._btn_mini_mode.setFixedHeight(28)
        self._btn_mini_mode.setToolTip("Collapse to a small always-on-top bar so the meeting window stays visible")
        self._btn_mini_mode.clicked.connect(self._toggle_mini_mode)
        control_bar.addWidget(self._btn_mini_mode)

        control_bar.addStretch()

        # FPS readout
        fps_container = QHBoxLayout()
        fps_container.setSpacing(4)
        fps_icon = QLabel("FPS")
        fps_icon.setStyleSheet("font-size: 10px; color: #666666;")
        self._fps_label = QLabel("—")
        self._fps_label.setStyleSheet(
            "font-size: 11px; font-weight: 500; color: #aaaaaa; min-width: 24px;"
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
        faces_icon.setStyleSheet("font-size: 10px; color: #666666;")
        self._faces_label = QLabel("—")
        self._faces_label.setStyleSheet(
            "font-size: 11px; font-weight: 500; color: #aaaaaa; min-width: 18px;"
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

    def _build_mini_bar(self) -> QWidget:
        """Compact status + Start/Stop/Exit row, shown only in Mini mode."""
        bar = QWidget()
        bar.setObjectName("card")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(8)

        self._mini_status_dot = QLabel("●")
        self._mini_status_dot.setStyleSheet("color: #6B7280; font-size: 10px;")
        layout.addWidget(self._mini_status_dot)

        self._mini_status_label = QLabel("Idle")
        self._mini_status_label.setStyleSheet("color: #9CA3AF; font-size: 11px;")
        layout.addWidget(self._mini_status_label)
        layout.addStretch()

        # The shared #btn_start/#btn_stop/#btn_screenshot rules use
        # padding meant for full-width buttons (e.g. "6px 16px"), which
        # would swallow all the space in these small fixed-size buttons
        # and hide the glyph entirely — override padding down to fit.
        mini_btn_style = "padding: 2px;"

        self._btn_mini_start = QPushButton("▶")
        self._btn_mini_start.setObjectName("btn_start")
        self._btn_mini_start.setFixedSize(32, 26)
        self._btn_mini_start.setStyleSheet(mini_btn_style)
        self._btn_mini_start.setToolTip("Start detection")
        self._btn_mini_start.clicked.connect(self._on_start)
        layout.addWidget(self._btn_mini_start)

        self._btn_mini_stop = QPushButton("◻")
        self._btn_mini_stop.setObjectName("btn_stop")
        self._btn_mini_stop.setFixedSize(32, 26)
        self._btn_mini_stop.setStyleSheet(mini_btn_style)
        self._btn_mini_stop.setEnabled(False)
        self._btn_mini_stop.setToolTip("Stop detection")
        self._btn_mini_stop.clicked.connect(self._on_stop)
        layout.addWidget(self._btn_mini_stop)

        btn_exit_mini = QPushButton("⤢")
        btn_exit_mini.setObjectName("btn_screenshot")
        btn_exit_mini.setFixedSize(32, 26)
        btn_exit_mini.setStyleSheet(mini_btn_style)
        btn_exit_mini.setToolTip("Exit Mini Mode")
        btn_exit_mini.clicked.connect(self._toggle_mini_mode)
        layout.addWidget(btn_exit_mini)

        bar.setVisible(False)
        return bar

    def _build_mini_face_strip(self) -> QWidget:
        """
        Card showing live, actual face-crop thumbnails (up to MAX_FACES) for
        Mini mode — the whole point of the app is visual verification, so
        Mini mode must let the user SEE which face a score belongs to, not
        just an abstract percentage. Replaces the confidence gauge as the
        primary visual in Mini mode (see _enter_mini_mode/_exit_mini_mode).
        """
        card = QWidget()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        title = QLabel("LIVE FACES")
        title.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;")
        layout.addWidget(title)

        self._mini_faces_empty_label = QLabel("No face detected")
        self._mini_faces_empty_label.setStyleSheet("color: #6B7280; font-size: 11px; font-style: italic;")
        layout.addWidget(self._mini_faces_empty_label)

        strip_layout = QHBoxLayout()
        strip_layout.setSpacing(8)
        self._mini_thumbnails = [FaceThumbnail(size=64) for _ in range(config.MAX_FACES)]
        for thumb in self._mini_thumbnails:
            strip_layout.addWidget(thumb)
        strip_layout.addStretch()
        layout.addLayout(strip_layout)

        card.setVisible(False)
        return card

    def _build_right_panel(self) -> QWidget:
        """Build right panel with info cards."""
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(8)

        # Mini Mode status/control bar (hidden unless in Mini mode)
        self._mini_bar = self._build_mini_bar()
        right_layout.addWidget(self._mini_bar)

        # Mini Mode live face thumbnails (hidden unless in Mini mode)
        self._mini_face_strip = self._build_mini_face_strip()
        right_layout.addWidget(self._mini_face_strip)

        # Confidence Card
        self._confidence_card = self._build_confidence_card()
        right_layout.addWidget(self._confidence_card)

        # Detection Analysis Card — always-visible reasoning for the latest result
        self._detection_analysis_card = self._build_detection_analysis_card()
        right_layout.addWidget(self._detection_analysis_card)

        # Alert Card (hidden by default)
        self._alert_card = self._build_alert_card()
        self._alert_card.setVisible(False)
        right_layout.addWidget(self._alert_card)

        # Session Info Card
        self._session_info_card = self._build_session_info_card()
        right_layout.addWidget(self._session_info_card)

        # System Log Card
        self._log_card = QWidget()
        log_card = self._log_card
        log_card.setObjectName("card")
        log_card_layout = QVBoxLayout(log_card)
        log_card_layout.setContentsMargins(8, 9, 8, 8)
        log_card_layout.setSpacing(5)

        log_header = QHBoxLayout()
        log_title = QLabel("SYSTEM LOG")
        log_title.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: 600; letter-spacing: 0.5px;")
        log_header.addWidget(log_title)
        log_header.addStretch()
        self._btn_clear_log = QPushButton("Clear")
        self._btn_clear_log.setObjectName("btn_clear_log")
        self._btn_clear_log.setFixedHeight(18)
        self._btn_clear_log.clicked.connect(self._on_clear_log)
        log_header.addWidget(self._btn_clear_log)
        log_card_layout.addLayout(log_header)

        self._session_log = SessionLog()
        self._session_log.setObjectName("session_log")
        log_card_layout.addWidget(self._session_log, stretch=1)

        log_card.setMinimumHeight(130)
        right_layout.addWidget(log_card, stretch=1)

        # Card count is variable (alert card toggles, analysis reasons vary
        # 0-3 lines), so total height can exceed the window's minimum size.
        # Scroll rather than let every card get squeezed below a usable size.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(right_container)
        return scroll

    def _build_confidence_card(self) -> QWidget:
        """Build confidence score card."""
        card = QWidget()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 9, 8, 6)
        layout.setSpacing(3)

        # Title row + info icon
        title_row = QHBoxLayout()
        title = QLabel("CONFIDENCE SCORE")
        title.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: 600; letter-spacing: 0.5px;")
        title_row.addWidget(title)
        title_row.addStretch()
        info_icon = QLabel("ⓘ")
        info_icon.setStyleSheet("color: #4B5563; font-size: 11px;")
        info_icon.setToolTip(
            "Confidence that the most suspicious face currently on screen is a deepfake, "
            "from the BitMind API's raw score (not smoothed)."
        )
        title_row.addWidget(info_icon)
        layout.addLayout(title_row)

        # Semicircular gauge
        self._gauge = GaugeWidget()
        layout.addWidget(self._gauge)

        return card

    _ANALYSIS_MAX_REASONS = 3

    def _build_detection_analysis_card(self) -> QWidget:
        """Build the Detection Analysis card: verdict + full bullet-point reasoning for the latest result."""
        card = QWidget()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 9, 8, 9)
        layout.setSpacing(6)

        # Title
        title_row = QHBoxLayout()
        icon = QLabel("🔍")
        icon.setStyleSheet("font-size: 11px;")
        title_row.addWidget(icon)
        title = QLabel("DETECTION ANALYSIS")
        title.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: 600; letter-spacing: 0.5px;")
        title_row.addWidget(title)
        title_row.addStretch()
        layout.addLayout(title_row)

        # Verdict row: colour-coded pill + confidence % + relative timestamp
        verdict_row = QHBoxLayout()
        verdict_row.setSpacing(6)
        self._analysis_verdict_badge = QLabel("")
        self._analysis_verdict_badge.setFixedHeight(18)
        verdict_row.addWidget(self._analysis_verdict_badge)
        self._analysis_pct_label = QLabel("")
        self._analysis_pct_label.setStyleSheet("color: #E5E7EB; font-size: 12px; font-weight: bold;")
        verdict_row.addWidget(self._analysis_pct_label)
        verdict_row.addStretch()
        self._analysis_time_label = QLabel("")
        self._analysis_time_label.setStyleSheet("color: #6B7280; font-size: 9px;")
        verdict_row.addWidget(self._analysis_time_label)
        layout.addLayout(verdict_row)

        # Empty-state placeholder, shown until the first result arrives
        self._analysis_empty_label = QLabel("No analysis yet — start detection to see results.")
        self._analysis_empty_label.setWordWrap(True)
        self._analysis_empty_label.setStyleSheet("color: #6B7280; font-size: 10px; font-style: italic;")
        layout.addWidget(self._analysis_empty_label)

        # Fixed pool of bullet rows (max reasons a DEEPFAKE result can carry
        # is 3 — see runtime/detection/explainer.py). Reused/hidden rather
        # than rebuilt each update to avoid widget churn at frame rate.
        self._analysis_bullet_rows = []
        for _ in range(self._ANALYSIS_MAX_REASONS):
            row_layout = QHBoxLayout()
            row_layout.setSpacing(5)
            bullet_icon = QLabel("●")
            bullet_icon.setStyleSheet("color: #4B5563; font-size: 7px;")
            bullet_icon.setFixedWidth(9)
            bullet_text = QLabel("")
            bullet_text.setWordWrap(True)
            bullet_text.setStyleSheet("color: #C4C9D4; font-size: 10px;")
            row_layout.addWidget(bullet_icon, 0, Qt.AlignmentFlag.AlignTop)
            row_layout.addWidget(bullet_text, 1)
            layout.addLayout(row_layout)
            self._analysis_bullet_rows.append((bullet_icon, bullet_text))

        self._update_detection_analysis(None)
        return card

    def _update_detection_analysis(self, result):
        """
        Sync the Detection Analysis card with a FaceResult (or None to show
        the empty state). Called every frame from _on_frame_ready, mirroring
        the Confidence Score gauge so both always reflect the same result.
        """
        if result is None:
            self._analysis_key = None
            self._analysis_empty_label.setVisible(True)
            self._analysis_verdict_badge.setVisible(False)
            self._analysis_pct_label.setVisible(False)
            self._analysis_time_label.setVisible(False)
            for bullet_icon, bullet_text in self._analysis_bullet_rows:
                bullet_icon.setVisible(False)
                bullet_text.setVisible(False)
            return

        elapsed = time.perf_counter() - result.last_updated
        age_str = "just now" if elapsed < 1.0 else f"{elapsed:.0f}s ago"
        self._analysis_time_label.setText(age_str)

        key = (result.face_id, result.last_updated)
        if key == self._analysis_key:
            return
        self._analysis_key = key

        self._analysis_empty_label.setVisible(False)
        self._analysis_verdict_badge.setVisible(True)
        self._analysis_pct_label.setVisible(True)
        self._analysis_time_label.setVisible(True)

        is_fake = result.risk_level == "DEEPFAKE"
        badge_bg, badge_fg = ("#EF4444", "#2D0A0A") if is_fake else ("#22C55E", "#052E16")
        self._analysis_verdict_badge.setText(f"  {result.risk_level}  ")
        self._analysis_verdict_badge.setStyleSheet(
            f"background-color: {badge_bg}; color: {badge_fg}; font-size: 9px; "
            f"font-weight: bold; border-radius: 9px;"
        )
        self._analysis_pct_label.setText(f"{result.p_fake_raw * 100:.1f}%")

        reasons = result.explanation or [
            "No local visual artifacts detected by heuristic analysis — "
            "this result is based on the cloud model's own assessment."
        ]
        for i, (bullet_icon, bullet_text) in enumerate(self._analysis_bullet_rows):
            show = i < len(reasons)
            bullet_icon.setVisible(show)
            bullet_text.setVisible(show)
            if show:
                bullet_text.setText(reasons[i])

    def _build_alert_card(self) -> QWidget:
        """Build alert card (shown when alert triggered)."""
        card = QWidget()
        card.setObjectName("alert_card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 9, 8, 9)
        layout.setSpacing(8)

        # Header
        header_layout = QHBoxLayout()
        icon_label = QLabel("⚠")
        icon_label.setStyleSheet("color: #F59E0B; font-size: 13px;")
        header_label = QLabel("Deepfake detected")
        header_label.setStyleSheet(
            "color: #FBBF24; font-size: 11px; font-weight: bold;"
        )
        header_layout.addWidget(icon_label)
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Full reasoning lives in the always-visible Detection Analysis card
        # above this one — this banner is just the dismissible action prompt.
        hint_label = QLabel("See Detection Analysis above for the full reasoning.")
        hint_label.setWordWrap(True)
        hint_label.setStyleSheet("color: #D8B675; font-size: 10px;")
        layout.addWidget(hint_label)

        # Acknowledge button
        self._btn_acknowledge = QPushButton("Acknowledge alert")
        self._btn_acknowledge.setObjectName("btn_acknowledge")
        self._btn_acknowledge.clicked.connect(self._on_acknowledge_alert)
        layout.addWidget(self._btn_acknowledge)

        return card

    @staticmethod
    def _build_info_row(icon: str, label_text: str) -> tuple:
        """Build an [icon] Label ... Value row for the session info card. Returns (layout, value_label)."""
        row_layout = QHBoxLayout()
        row_layout.setContentsMargins(0, 5, 0, 5)
        row_layout.setSpacing(6)
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("color: #4B5563; font-size: 11px;")
        icon_label.setFixedWidth(14)
        row_layout.addWidget(icon_label)
        label = QLabel(label_text)
        label.setStyleSheet("color: #888; font-size: 10px;")
        row_layout.addWidget(label)
        row_layout.addStretch()
        value = QLabel("")
        value.setStyleSheet("color: #E5E7EB; font-size: 10px;")
        row_layout.addWidget(value)
        return row_layout, value

    def _build_session_info_card(self) -> QWidget:
        """Build session info card."""
        card = QWidget()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(8, 9, 8, 9)
        layout.setSpacing(0)

        # Title
        title = QLabel("SESSION INFO")
        title.setStyleSheet("color: #6B7280; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 4px;")
        layout.addWidget(title)

        status_row, self._status_value = self._build_info_row("👤", "Status")
        self._status_value.setText("● Active")
        layout.addLayout(status_row)

        duration_row, self._duration_value = self._build_info_row("◷", "Duration")
        self._duration_value.setText("00:00:00")
        layout.addLayout(duration_row)

        frames_row, self._frames_value = self._build_info_row("▥", "Frames")
        self._frames_value.setText("0")
        layout.addLayout(frames_row)

        model_row, model_value = self._build_info_row("▣", "Model")
        model_value.setText(config.MODEL_NAME.replace("_", "-").title())
        layout.addLayout(model_row)

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
        table.setFixedHeight(120)

        return table

    def _update_status_bar(self):
        """Update status bar with current state."""
        status_bar = self.statusBar()
        status_bar.clearMessage()
        status_bar.setStyleSheet(
            "background-color: #0B0E14; color: #6B7280; font-size: 10px; border-top: 1px solid #1F2430;"
        )

        def _sep() -> QLabel:
            lbl = QLabel("│")
            lbl.setStyleSheet("color: #2A2E38;")
            return lbl

        lbl_source = QLabel("🎥  Source: Zoom")
        lbl_source.setStyleSheet("color: #6B7280;")

        lbl_model = QLabel("✓  Model loaded")
        lbl_model.setStyleSheet("color: #22C55E;")

        lbl_db = QLabel("✓  DB connected")
        lbl_db.setStyleSheet("color: #22C55E;")

        lbl_threshold = QLabel(f"◎  Threshold: {config.FAKE_THRESHOLD * 100:.0f}%")
        lbl_threshold.setStyleSheet("color: #6B7280;")

        status_bar.addPermanentWidget(lbl_source)
        status_bar.addPermanentWidget(_sep())
        status_bar.addPermanentWidget(lbl_model)
        status_bar.addPermanentWidget(_sep())
        status_bar.addPermanentWidget(lbl_db)
        status_bar.addPermanentWidget(_sep())
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
                    padding: 1px 6px;
                    font-size: 10px;
                    font-weight: 500;
                """
                )
            else:
                badge.setStyleSheet(
                    """
                    background-color: #EAF3DE;
                    color: #3B6D11;
                    border-radius: 3px;
                    padding: 1px 6px;
                    font-size: 10px;
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

        state: one of "Idle", "Initializing", "Detecting", "Alert", "Degraded", "Stopping".
        """
        configs = {
            "Idle":         ("#6B7280", "#9CA3AF", "Idle"),
            "Initializing": ("#F59E0B", "#FBBF24", "Initializing…"),
            "Detecting":    ("#22C55E", "#4ADE80", "Active"),
            "Alert":        ("#EF4444", "#F87171", "Alert"),
            "Degraded":     ("#F59E0B", "#FBBF24", "Degraded — Reconnecting"),
            "Stopping":     ("#F59E0B", "#FBBF24", "Stopping…"),
        }
        dot_color, text_color, label = configs.get(state, configs["Idle"])
        self._status_state = state
        self._status_value.setText(
            f"<span style='color:{dot_color};'>●</span> "
            f"<span style='color:{text_color};'>{label}</span>"
        )
        self._status_value.setStyleSheet("font-size: 10px;")

        # Keep the Mini mode bar's status readout in sync too
        self._mini_status_dot.setStyleSheet(f"color: {dot_color}; font-size: 10px;")
        self._mini_status_label.setText(label)
        self._mini_status_label.setStyleSheet(f"color: {text_color}; font-size: 11px;")

    @pyqtSlot()
    def _on_start(self):
        """Start detection and alert threads."""
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_mini_start.setEnabled(False)
        self._btn_mini_stop.setEnabled(True)
        self._session_active = True
        self._session_start_time = 0
        self._frame_count = 0
        self._current_alert_active = False
        self._alert_count = 0
        self._degraded = False

        # Hide alert card
        self._alert_card.setVisible(False)

        # Reset confidence + analysis
        self._gauge.set_value(0, "—")
        self._update_detection_analysis(None)
        self._update_mini_face_strip([])

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
        self._detection_worker.low_light_warning.connect(self._on_low_light_warning)
        self._detection_worker.face_count_update.connect(self._on_face_count_update)
        self._detection_worker.degraded_mode.connect(self._on_degraded_mode)

        # Alert thread
        self._alert_thread = AlertThread()
        self._alert_thread.alert_triggered.connect(self._on_alert_triggered)
        self._alert_thread.alert_dismissed.connect(self._on_alert_dismissed)

        # Wire detection results to alert thread. A face being manually
        # re-checked keeps its previous verdict here, so re-checking a
        # deepfake doesn't auto-dismiss the alert and then re-raise it.
        self._detection_worker.frame_ready.connect(
            lambda frame, results, pending, rechecking: self._alert_thread.push_results(results + rechecking)
        )

        self._alert_thread.start()
        self._detection_worker.start()

        self._session_log_event("SESSION", "Detection started.")
        self._session_start_id = db_log_event("SESSION", "Detection started.")

        # Nudge the window to a screen edge so it's less likely to default
        # to a position that overlaps the meeting window. This app doesn't
        # track the meeting window's own bounds (only its own, for capture
        # self-exclusion), so this docks to the nearest free screen edge
        # rather than to the meeting window specifically.
        self._dock_to_screen_edge()
        if not self._mini_mode:
            self._session_log_event(
                "INFO", "Tip: use Mini Mode to keep the meeting fully visible while monitoring."
            )

    @pyqtSlot()
    def _on_stop(self):
        """Stop detection and alert threads, then prompt to save or discard."""
        # Show "Stopping…" immediately and force a repaint before any blocking work
        self._set_status("Stopping")
        QApplication.processEvents()

        self._btn_stop.setEnabled(False)
        self._btn_mini_stop.setEnabled(False)
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
        self._btn_mini_start.setEnabled(True)
        self._alert_card.setVisible(False)
        self._low_light_bar.setVisible(False)
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
            elif self._session_start_id is not None:
                # Discard: delete everything logged since the start event, so
                # no orphaned start or CLASSIFICATION rows are left behind.
                delete_events_from(self._session_start_id)

        # Return UI to Idle state and reset stats readouts
        self._set_status("Idle")
        self._fps_label.setText("—")
        self._faces_label.setText("—")

    @pyqtSlot()
    def _on_screenshot(self):
        """
        Save a full-screen PNG (this app's own window and its detection
        overlay included, plus whatever else is on screen) to the
        screenshots folder — evidence of what was actually seen, not just
        the video feed. Briefly lifts this window's capture exclusion,
        under capture_affinity_lock so DetectionWorker's live capture loop
        can never overlap that window and pick up the un-excluded app.
        """
        if self._video_widget.grab_screenshot() is None:
            self._session_log_event("INFO", "Screenshot skipped: no active feed.")
            return

        hwnd = int(self.winId())
        with capture_affinity_lock:
            cleared = self._capture_affinity_value is not None and clear_capture_exclusion(hwnd)
            try:
                with mss.mss() as sct:
                    monitor = sct.monitors[1]  # primary monitor — same scope DetectionWorker defaults to
                    shot = np.asarray(sct.grab(monitor))[:, :, :3]  # BGRA -> BGR
            finally:
                if cleared:
                    restore_capture_exclusion(hwnd, self._capture_affinity_value)

        screenshots_dir = os.path.join(config.BASE_DIR, "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        path = os.path.join(screenshots_dir, filename)
        cv2.imwrite(path, shot)
        self._session_log_event("INFO", f"Screenshot saved: {filename}")

    @pyqtSlot()
    def _on_acknowledge_alert(self):
        """Acknowledge the current alert."""
        self._alert_card.setVisible(False)
        self._current_alert_active = False
        if self._session_active and not self._degraded:
            self._set_status("Detecting")
        self._session_log_event("DISMISS", "Alert acknowledged.")
        db_log_event("DISMISS", "Alert acknowledged.")

    # -- Frame & Result Handling -----------------------------------------------

    @pyqtSlot(object, object, object, object)
    def _on_frame_ready(self, frame: np.ndarray, results: list, pending_bboxes: list, rechecking_results: list):
        """Handle incoming frame and detection results."""
        self._frame_count += 1
        self._frames_value.setText(f"{self._frame_count:,}")

        # First frame after start transitions Initializing -> Detecting
        if self._status_state == "Initializing":
            self._set_status("Detecting")

        # A face being re-checked still has its previous verdict, which stays
        # the latest known one until the fresh result lands.
        verdicts = results + rechecking_results

        # Update confidence + analysis based on worst result. Alerts are
        # raised only by AlertThread, so its debounce and Acknowledge hold.
        if verdicts:
            worst = max(verdicts, key=lambda r: r.p_fake_raw)
            # Use raw score for accuracy — real-time API response
            display_confidence = worst.p_fake_raw * 100
            self._gauge.set_value(display_confidence, worst.risk_level.title())
            self._update_detection_analysis(worst)
        else:
            self._gauge.set_value(0, "—")
            self._update_detection_analysis(None)

        # Only spend time converting crops to QPixmaps when Mini mode can
        # actually show them — full mode already shows faces via the live
        # video overlay, so this would be wasted work there.
        if self._mini_mode:
            self._update_mini_face_strip(verdicts)

    def _update_mini_face_strip(self, results: list):
        """Update Mini mode's live face thumbnails, ranked by confidence (most suspicious first)."""
        ranked = sorted(results, key=lambda r: r.p_fake_raw, reverse=True)[:len(self._mini_thumbnails)]
        key = [(r.face_id, r.last_updated) for r in ranked]
        if key == self._mini_strip_key:
            return
        self._mini_strip_key = key

        if not ranked:
            self._mini_faces_empty_label.setVisible(True)
            for thumb in self._mini_thumbnails:
                thumb.clear()
            return

        self._mini_faces_empty_label.setVisible(False)
        for i, thumb in enumerate(self._mini_thumbnails):
            if i < len(ranked) and ranked[i].thumbnail is not None:
                thumb.set_face(ranked[i].thumbnail, ranked[i].risk_level, ranked[i].p_fake_raw)
            else:
                thumb.clear()

    @pyqtSlot(float)
    def _on_fps_update(self, fps: float):
        """Update the FPS readout in the control bar."""
        self._fps_label.setText(str(round(fps)))

    @pyqtSlot(float)
    def _on_api_response_update(self, api_ms: float):
        self._lbl_api_ms.setText(f"API: {api_ms:.0f} ms")

    @pyqtSlot(bool)
    def _on_low_light_warning(self, is_low_light: bool):
        if self._low_light_bar.isVisible() != is_low_light:
            self._low_light_bar.setVisible(is_low_light)

    @pyqtSlot(bool)
    def _on_degraded_mode(self, is_degraded: bool):
        """Reflect BitMind API connectivity issues without stopping the session."""
        self._degraded = is_degraded
        if is_degraded:
            self._set_status("Degraded")
            self._session_log_event("ERROR", "BitMind API unreachable — detection degraded.")
        elif self._session_active and not self._current_alert_active:
            self._set_status("Detecting")
            self._session_log_event("INFO", "BitMind API connection recovered.")

    @pyqtSlot(int, int)
    def _on_face_count_update(self, shown: int, total: int):
        if shown == 0:
            self._faces_label.setText("—")
            self._faces_label.setToolTip("")
        elif total > shown:
            self._faces_label.setText(f"{shown}/{total}")
            self._faces_label.setToolTip(
                f"Analyzing {shown} of {total} detected faces (max {config.MAX_FACES})"
            )
        else:
            self._faces_label.setText(str(shown))
            self._faces_label.setToolTip("")

    @pyqtSlot(float, list)
    def _on_alert_triggered(self, peak_p_fake: float, explanation: list):
        """
        Handle alert trigger from alert thread — the single source of alerts.

        The log only records a one-line summary — the full reasoning
        (`explanation`) is surfaced continuously in the Detection Analysis
        card via _on_frame_ready, not duplicated into the log here.
        """
        self._alert_count += 1
        self._alert_card.setVisible(True)
        self._current_alert_active = True
        self._set_status("Alert")
        notify_alert(self, peak_p_fake)
        detail = f"Deepfake detected — Peak confidence: {peak_p_fake:.1%}"
        self._session_log_event("ALERT", detail)
        db_log_event("ALERT", detail, peak_confidence=peak_p_fake)

    @pyqtSlot()
    def _on_alert_dismissed(self):
        """Handle alert dismissal from alert thread."""
        if not self._current_alert_active:
            return  # already acknowledged by the user — nothing left to dismiss
        self._alert_card.setVisible(False)
        self._current_alert_active = False
        if self._session_active and not self._degraded:
            self._set_status("Detecting")
        self._session_log_event("DISMISS", "Alert dismissed automatically.")
        db_log_event("DISMISS", "Alert dismissed automatically.")

    @pyqtSlot(str)
    def _on_error(self, error_msg: str):
        """Handle errors from detection worker."""
        self._session_log_event("ERROR", error_msg)
        db_log_event("ERROR", error_msg)
        self._on_stop()

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
    background-color: #0B0E14;
    color: #E5E7EB;
    font-family: 'Segoe UI', sans-serif;
    font-size: 12px;
}

QSplitter::handle {
    background-color: transparent;
}

/* Right sidebar scroll area */
QScrollArea, QScrollArea > QWidget > QWidget {
    background-color: transparent;
    border: none;
}

QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #262B36;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #333A48;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: transparent;
}

/* Buttons */
QPushButton#btn_start {
    background-color: #22C55E;
    border: 1px solid #22C55E;
    color: #05170C;
    border-radius: 14px;
    padding: 5px 14px;
    font-size: 11px;
    font-weight: bold;
}

QPushButton#btn_start:hover {
    background-color: #34D871;
}

QPushButton#btn_start:pressed {
    background-color: #16A34A;
}

QPushButton#btn_start:disabled {
    background-color: #1B3324;
    border-color: #1B3324;
    color: #4B5563;
}

QPushButton#btn_stop {
    background-color: #EF4444;
    border: 1px solid #EF4444;
    color: #200606;
    border-radius: 14px;
    padding: 5px 14px;
    font-size: 11px;
    font-weight: bold;
}

QPushButton#btn_stop:hover {
    background-color: #F45B5B;
}

QPushButton#btn_stop:pressed {
    background-color: #DC2626;
}

QPushButton#btn_stop:disabled {
    background-color: #331414;
    border-color: #331414;
    color: #4B5563;
}

QPushButton#btn_screenshot, QPushButton#btn_clear_zone {
    background-color: #12151C;
    border: 1px solid #262B36;
    color: #C4C9D4;
    border-radius: 14px;
    padding: 5px 14px;
    font-size: 11px;
}

QPushButton#btn_screenshot:hover, QPushButton#btn_clear_zone:hover {
    background-color: #1B1F29;
    border-color: #333A48;
}

QPushButton#btn_screenshot:pressed, QPushButton#btn_clear_zone:pressed {
    background-color: #21262E;
}

QPushButton#btn_clear_zone:disabled {
    background-color: #12151C;
    border: 1px solid #1B1F29;
    color: #3A4150;
}

QPushButton#btn_acknowledge {
    background-color: #F59E0B;
    border: 1px solid #F59E0B;
    color: #241A02;
    border-radius: 13px;
    padding: 5px 12px;
    font-size: 11px;
    font-weight: bold;
}

QPushButton#btn_acknowledge:hover {
    background-color: #FBB63A;
}

QPushButton#btn_clear_log, QPushButton#btn_delete_session, QPushButton#btn_clear_history {
    background-color: transparent;
    border: 1px solid #262B36;
    color: #8A93A3;
    border-radius: 10px;
    padding: 2px 10px;
    font-size: 11px;
}
QPushButton#btn_clear_log:hover {
    background-color: #1B1F29;
    border-color: #333A48;
    color: #E5E7EB;
}

QPushButton#btn_delete_session {
    border-color: #3A2020;
    color: #EF4444;
}
QPushButton#btn_delete_session:disabled {
    border-color: #1B1F29;
    color: #3A4150;
}
QPushButton#btn_delete_session:hover {
    background-color: #2A1414;
}

QPushButton#btn_clear_history:hover {
    background-color: #2A1414;
    border-color: #EF4444;
    color: #EF4444;
}

/* Cards */
QWidget#card {
    background-color: #12151C;
    border: 1px solid #1F2430;
    border-radius: 14px;
}

QWidget#video_frame {
    background-color: #000000;
    border: 1px solid #1F2430;
    border-radius: 14px;
}

QWidget#alert_card {
    background-color: #241B08;
    border: 1px solid #F59E0B;
    border-radius: 14px;
}

/* Table */
QTableWidget#history_table {
    background-color: #0B0E14;
    color: #C4C9D4;
    border: none;
    gridline-color: #1F2430;
    alternate-background-color: #0B0E14;
}

QTableWidget#history_table::item {
    padding: 4px 6px;
    border-bottom: 1px solid #1F2430;
}

QTableWidget#history_table::item:selected {
    background-color: #1B1F29;
}

QHeaderView::section {
    background-color: transparent;
    color: #6B7280;
    font-size: 10px;
    border: none;
    border-bottom: 1px solid #1F2430;
    padding: 3px 6px;
}

/* Status Bar */
QStatusBar {
    background-color: #0B0E14;
}

QStatusBar::item {
    border: none;
}

/* Video Widget */
QWidget#video_widget {
    background-color: #000000;
}

/* Session Log */
QTextEdit#session_log {
    background-color: transparent;
    color: #C4C9D4;
    border: none;
    font-family: 'Consolas', monospace;
    font-size: 10px;
    padding: 2px;
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

    @pyqtSlot(str)
    def _on_recheck_requested(self, face_id: str):
        """User clicked a face's "Re-check" button — run one more classification cycle for it."""
        if self._detection_worker is not None:
            self._detection_worker.request_recheck(face_id)

    @pyqtSlot()
    def _on_clear_log(self):
        self._session_log.clear()

    # -- Mini Mode ---------------------------------------------------------------

    _MINI_MODE_SIZE = (300, 220)
    _FULL_MODE_MIN_SIZE = (820, 600)

    @pyqtSlot()
    def _toggle_mini_mode(self):
        if self._mini_mode:
            self._exit_mini_mode()
        else:
            self._enter_mini_mode()

    def _enter_mini_mode(self):
        """
        Collapse to a small always-on-top bar showing status + live face
        thumbnails (not just an abstract gauge) so the user can actually
        see which face a score belongs to, plus Start/Stop — all while the
        meeting window underneath stays visible. Detection keeps running
        unaffected — hidden sections are only setVisible(False), never
        destroyed or rebuilt, so all state (Session History, Detection
        Analysis, System Log) is preserved and reappears exactly as it was
        on returning to Full mode.
        """
        if self._mini_mode:
            return
        self._full_mode_geometry = self.geometry()
        self._mini_mode = True

        self._left_panel.setVisible(False)
        self._confidence_card.setVisible(False)
        self._detection_analysis_card.setVisible(False)
        self._session_info_card.setVisible(False)
        self._log_card.setVisible(False)
        self._history_section.setVisible(False)
        self.statusBar().setVisible(False)
        self._mini_bar.setVisible(True)
        self._mini_face_strip.setVisible(True)
        self._btn_mini_mode.setText("⛶ Full Mode")

        # Changing window flags on an already-visible top-level window
        # needs a hide/show cycle on Windows for WindowStaysOnTopHint to
        # actually take effect.
        self.hide()
        self.setMinimumSize(*self._MINI_MODE_SIZE)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.show()
        self.resize(*self._MINI_MODE_SIZE)
        self._dock_to_screen_edge()

    def _exit_mini_mode(self):
        """Restore the full window layout, preserving all underlying state."""
        if not self._mini_mode:
            return
        self._mini_mode = False

        self._left_panel.setVisible(True)
        self._confidence_card.setVisible(True)
        self._detection_analysis_card.setVisible(True)
        self._session_info_card.setVisible(True)
        self._log_card.setVisible(True)
        if self._history_table.rowCount() > 0:
            self._history_section.setVisible(True)
        self.statusBar().setVisible(True)
        self._mini_bar.setVisible(False)
        self._mini_face_strip.setVisible(False)
        self._btn_mini_mode.setText("⛶ Mini Mode")

        self.hide()
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, False)
        self.setMinimumSize(*self._FULL_MODE_MIN_SIZE)
        self.show()
        if self._full_mode_geometry is not None:
            self.setGeometry(self._full_mode_geometry)

    def _dock_to_screen_edge(self):
        """
        Snap the window to the right edge of the primary screen's available
        work area (i.e. excluding the taskbar). This app does not track the
        meeting window's own bounds — only its own, for capture
        self-exclusion (see _window_screen_rect) — so this docks to a free
        screen edge as a sensible default rather than to the meeting
        window specifically.
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        w = min(self.width(), avail.width())
        h = min(self.height(), avail.height())
        if (w, h) != (self.width(), self.height()):
            self.resize(w, h)
        x = avail.right() - w + 1
        y = avail.top()
        self.move(max(avail.left(), x), y)

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

    def showEvent(self, event):
        super().showEvent(event)
        # winId() forces native window creation, so this must run after the
        # window is actually shown; guarded to apply only once per process.
        if not self._capture_exclusion_applied:
            self._capture_exclusion_applied = True
            mechanism, affinity = apply_capture_exclusion(int(self.winId()))
            self._capture_affinity_value = affinity
            self._session_log_event("INFO", f"Capture exclusion: {mechanism}")

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
