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
    QHeaderView,
    QAbstractItemView,
    QFrame,
    QDialog,
    QApplication,
)
from PyQt6.QtCore import pyqtSlot, Qt, QTimer, QRectF, QPointF
from PyQt6.QtGui import QColor, QFont, QIcon, QPixmap, QPainter, QPainterPath, QPen, QBrush
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from runtime.gui.video_widget import VideoWidget
from runtime.gui.session_log import SessionLog
from runtime.gui.dialogs import SaveSessionDialog
from runtime.gui.gauge_widget import GaugeWidget
from runtime.threads.detection_worker import DetectionWorker
from runtime.threads.alert_thread import AlertThread
from runtime.detection.face_detector import FaceDetector
from runtime.detection.inference import DeepfakeInferenceEngine
from runtime.storage.database import log_event as db_log_event, get_all_sessions, delete_session, clear_history
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
        
        # Main layout: padded so cards have breathing room from the window edge
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(20, 16, 20, 12)
        root_layout.setSpacing(14)

        # --- TOP SECTION: LEFT PANEL (video) + RIGHT PANEL (cards) ---
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setHandleWidth(14)

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
        left_layout.setSpacing(8)

        # "LIVE PREVIEW" section header with a small green status dot
        preview_header = QHBoxLayout()
        preview_header.setSpacing(6)
        live_dot = QLabel("●")
        live_dot.setStyleSheet("color: #22C55E; font-size: 9px;")
        preview_header.addWidget(live_dot)
        preview_label = QLabel("LIVE PREVIEW")
        preview_label.setStyleSheet(
            "color: #6B7280; font-size: 11px; font-weight: 600; letter-spacing: 1px;"
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
        self._video_widget.setMinimumHeight(300)
        video_frame_layout.addWidget(self._video_widget)
        left_layout.addWidget(video_frame, stretch=1)

        # Low-light warning bar (hidden until triggered)
        self._low_light_bar = QLabel("  Low light detected — detection accuracy may be reduced")
        self._low_light_bar.setStyleSheet(
            "background-color: #5C3500; color: #FFCC66; font-size: 11px; padding: 3px 8px; border-radius: 6px;"
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

        # System Log Card
        log_card = QWidget()
        log_card.setObjectName("card")
        log_card_layout = QVBoxLayout(log_card)
        log_card_layout.setContentsMargins(10, 12, 10, 10)
        log_card_layout.setSpacing(6)

        log_header = QHBoxLayout()
        log_title = QLabel("SYSTEM LOG")
        log_title.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;")
        log_header.addWidget(log_title)
        log_header.addStretch()
        self._btn_clear_log = QPushButton("Clear")
        self._btn_clear_log.setObjectName("btn_clear_log")
        self._btn_clear_log.setFixedHeight(20)
        self._btn_clear_log.clicked.connect(self._on_clear_log)
        log_header.addWidget(self._btn_clear_log)
        log_card_layout.addLayout(log_header)

        self._session_log = SessionLog()
        self._session_log.setObjectName("session_log")
        log_card_layout.addWidget(self._session_log, stretch=1)

        right_layout.addWidget(log_card, stretch=1)

        return right_container

    def _build_confidence_card(self) -> QWidget:
        """Build confidence score card."""
        card = QWidget()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 12, 10, 8)
        layout.setSpacing(4)

        # Title row + info icon
        title_row = QHBoxLayout()
        title = QLabel("CONFIDENCE SCORE")
        title.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;")
        title_row.addWidget(title)
        title_row.addStretch()
        info_icon = QLabel("ⓘ")
        info_icon.setStyleSheet("color: #4B5563; font-size: 12px;")
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
        icon_label.setStyleSheet("color: #F59E0B; font-size: 14px;")
        header_label = QLabel("Deepfake detected")
        header_label.setStyleSheet(
            "color: #FBBF24; font-size: 12px; font-weight: bold;"
        )
        header_layout.addWidget(icon_label)
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Explanation text (why this face was flagged). Single-line + elided:
        # the sidebar's fixed-height layout can't grow to fit wrapped
        # multi-sentence text without clipping, so the full text goes in the
        # tooltip and session log instead (see _show_alert).
        self._alert_reason_label = QLabel("")
        self._alert_reason_label.setWordWrap(False)
        self._alert_reason_label.setStyleSheet("color: #D8B675; font-size: 11px;")
        layout.addWidget(self._alert_reason_label)

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
        row_layout.setContentsMargins(0, 7, 0, 7)
        row_layout.setSpacing(8)
        icon_label = QLabel(icon)
        icon_label.setStyleSheet("color: #4B5563; font-size: 12px;")
        icon_label.setFixedWidth(16)
        row_layout.addWidget(icon_label)
        label = QLabel(label_text)
        label.setStyleSheet("color: #888; font-size: 11px;")
        row_layout.addWidget(label)
        row_layout.addStretch()
        value = QLabel("")
        value.setStyleSheet("color: #E5E7EB; font-size: 11px;")
        row_layout.addWidget(value)
        return row_layout, value

    def _build_session_info_card(self) -> QWidget:
        """Build session info card."""
        card = QWidget()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(0)

        # Title
        title = QLabel("SESSION INFO")
        title.setStyleSheet("color: #6B7280; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; margin-bottom: 6px;")
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
        table.setFixedHeight(150)

        return table

    def _update_status_bar(self):
        """Update status bar with current state."""
        status_bar = self.statusBar()
        status_bar.clearMessage()
        status_bar.setStyleSheet(
            "background-color: #0B0E14; color: #6B7280; font-size: 11px; border-top: 1px solid #1F2430;"
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
            "Idle":         ("#6B7280", "#9CA3AF", "Idle"),
            "Initializing": ("#F59E0B", "#FBBF24", "Initializing…"),
            "Detecting":    ("#22C55E", "#4ADE80", "Active"),
            "Alert":        ("#EF4444", "#F87171", "Alert"),
            "Stopping":     ("#F59E0B", "#FBBF24", "Stopping…"),
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
        self._alert_reason_label.setText("")

        # Reset confidence
        self._gauge.set_value(0, "—")

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

        # Alert thread
        self._alert_thread = AlertThread()
        self._alert_thread.alert_triggered.connect(self._on_alert_triggered)
        self._alert_thread.alert_dismissed.connect(self._on_alert_dismissed)

        # Wire detection results to alert thread
        self._detection_worker.frame_ready.connect(
            lambda frame, results, pending: self._alert_thread.push_results(results)
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

    @pyqtSlot(object, object, object)
    def _on_frame_ready(self, frame: np.ndarray, results: list, pending_bboxes: list):
        """Handle incoming frame and detection results."""
        self._frame_count += 1
        self._frames_value.setText(f"{self._frame_count:,}")

        # First frame after start transitions Initializing -> Detecting
        if self._status_state == "Initializing":
            self._set_status("Detecting")

        # Update confidence based on worst result
        if results:
            worst = max(results, key=lambda r: r.p_fake_raw)
            # Use raw score for accuracy — real-time API response
            display_confidence = worst.p_fake_raw * 100
            self._gauge.set_value(display_confidence, worst.risk_level.title())

            if worst.p_fake_raw >= config.FAKE_THRESHOLD and not self._current_alert_active:
                self._show_alert(worst.explanation)
        else:
            self._gauge.set_value(0, "—")

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
        """Handle alert trigger from alert thread."""
        if not self._current_alert_active:
            self._show_alert(explanation)
        reason_suffix = f" Reasons: {'; '.join(explanation)}" if explanation else ""
        detail = f"Deepfake detected. Peak confidence: {peak_p_fake:.1%}{reason_suffix}"
        self._session_log_event("ALERT", detail)
        db_log_event("ALERT", detail, peak_confidence=peak_p_fake)

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

    def _show_alert(self, explanation: list = None):
        """Show alert card and reflect the Alert state."""
        if not self._current_alert_active:
            self._alert_count += 1
        reasons = explanation or []
        text = " · ".join(reasons) if reasons else (
            "No local visual artifacts detected by heuristic analysis — "
            "this result is based on the cloud model's own assessment."
        )
        # Elide to fit the sidebar so the fixed-height layout never has to
        # grow (which would clip against the button below it). Measured off
        # the alert card's parent container rather than the label itself:
        # the card is hidden until now, so its own width may not yet reflect
        # the real layout-constrained size, but the always-visible parent's
        # width is reliable. The full text is always available via tooltip
        # and is also written to the session log / DB detail in full.
        parent_width = self._alert_card.parentWidget().width() or 300
        available_width = max(parent_width - 40, 150)
        display_text = self._alert_reason_label.fontMetrics().elidedText(
            text, Qt.TextElideMode.ElideRight, available_width
        )
        self._alert_reason_label.setText(display_text)
        self._alert_reason_label.setToolTip(text)
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
    background-color: #0B0E14;
    color: #E5E7EB;
    font-family: 'Segoe UI', sans-serif;
    font-size: 13px;
}

QSplitter::handle {
    background-color: transparent;
}

/* Buttons */
QPushButton#btn_start {
    background-color: #22C55E;
    border: 1px solid #22C55E;
    color: #05170C;
    border-radius: 16px;
    padding: 6px 16px;
    font-size: 12px;
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
    border-radius: 16px;
    padding: 6px 16px;
    font-size: 12px;
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
    border-radius: 16px;
    padding: 6px 16px;
    font-size: 12px;
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
    border-radius: 14px;
    padding: 6px 14px;
    font-size: 12px;
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
    padding: 6px 8px;
    border-bottom: 1px solid #1F2430;
}

QTableWidget#history_table::item:selected {
    background-color: #1B1F29;
}

QHeaderView::section {
    background-color: transparent;
    color: #6B7280;
    font-size: 11px;
    border: none;
    border-bottom: 1px solid #1F2430;
    padding: 4px 8px;
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
    font-size: 11px;
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

    @pyqtSlot()
    def _on_clear_log(self):
        self._session_log.clear()

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
