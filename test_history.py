"""Tests for session-history storage (database.py) and the MainWindow history table, against a temporary DB."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QLabel

import config
from runtime.storage import database
from runtime.gui.main_window import MainWindow


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """Point every database call at an empty per-test SQLite file instead of session_log.db."""
    monkeypatch.setattr(config, "DB_PATH", str(tmp_path / "test.db"))


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    # Module-scoped because MainWindow construction is slow; the DB is patched
    # during construction too so its initial refresh_history() never reads the real DB.
    app = QApplication.instance() or QApplication([])
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(config, "DB_PATH", str(tmp_path_factory.mktemp("db") / "init.db"))
        win = MainWindow()
    yield win
    win.close()
    app.processEvents()


def _insert(timestamp: str, event_type: str, detail: str = ""):
    """Insert an event with an explicit timestamp, since log_event() always stamps 'now'."""
    conn = database._get_connection()
    conn.execute(
        "INSERT INTO session_events (timestamp, event_type, detail) VALUES (?, ?, ?)",
        (timestamp, event_type, detail),
    )
    conn.commit()
    conn.close()


def _insert_two_sessions():
    _insert("2026-05-06 09:00:00", "SESSION", "Detection started.")
    _insert("2026-05-06 09:03:15", "SESSION", "Detection stopped.")
    _insert("2026-05-06 10:00:00", "SESSION", "Detection started.")
    _insert("2026-05-06 10:01:00", "ALERT", "Deepfake alert")
    _insert("2026-05-06 10:02:00", "ALERT", "Deepfake alert")
    _insert("2026-05-06 10:05:30", "SESSION", "Detection stopped.")


def _badge_text(win: MainWindow, row: int) -> str:
    return win._history_table.cellWidget(row, 3).findChild(QLabel).text()


def test_log_event_round_trip():
    database.log_event("ALERT", "Deepfake alert", peak_confidence=0.91, api_response_ms=420.0)
    events = database.get_session_history()
    assert len(events) == 1
    assert events[0]["event_type"] == "ALERT"
    assert events[0]["detail"] == "Deepfake alert"
    assert events[0]["peak_confidence"] == pytest.approx(0.91)
    assert events[0]["api_response_ms"] == pytest.approx(420.0)


def test_sessions_reconstructed_newest_first_with_duration_and_alerts():
    _insert_two_sessions()
    assert database.get_all_sessions() == [
        {"started_at": "2026-05-06 10:00:00", "duration": "00:05:30",
         "alert_count": 2, "result": "Deepfake detected"},
        {"started_at": "2026-05-06 09:00:00", "duration": "00:03:15",
         "alert_count": 0, "result": "Clean"},
    ]


def test_unfinished_session_is_skipped():
    _insert("2026-05-06 09:00:00", "SESSION", "Detection started.")
    _insert("2026-05-06 09:01:00", "ALERT", "Deepfake alert")
    assert database.get_all_sessions() == []


def test_delete_session_removes_only_that_session():
    _insert_two_sessions()
    database.delete_session("2026-05-06 10:00:00")
    sessions = database.get_all_sessions()
    assert [s["started_at"] for s in sessions] == ["2026-05-06 09:00:00"]
    assert not any(e["event_type"] == "ALERT" for e in database.get_session_history())


def test_orphaned_start_does_not_claim_next_session_stop():
    _insert("2026-05-06 08:00:00", "SESSION", "Detection started.")
    _insert("2026-05-06 08:01:00", "ALERT", "Deepfake alert")
    _insert_two_sessions()
    assert [s["started_at"] for s in database.get_all_sessions()] == [
        "2026-05-06 10:00:00", "2026-05-06 09:00:00",
    ]


def test_delete_session_of_orphaned_start_keeps_next_session():
    _insert("2026-05-06 08:00:00", "SESSION", "Detection started.")
    _insert_two_sessions()
    database.delete_session("2026-05-06 08:00:00")
    assert len(database.get_all_sessions()) == 2


def test_log_event_returns_row_id():
    first = database.log_event("INFO", "one")
    second = database.log_event("INFO", "two")
    assert second == first + 1


def test_delete_events_from_discards_session():
    _insert_two_sessions()
    start_id = database.log_event("SESSION", "Detection started.")
    database.log_event("CLASSIFICATION", "face=f0 risk=DEEPFAKE trigger=auto", peak_confidence=0.9)
    database.log_event("ALERT", "Deepfake alert")
    database.delete_events_from(start_id)
    # Only the 6 rows of the two saved sessions remain.
    assert len(database.get_session_history()) == 6
    assert len(database.get_all_sessions()) == 2


def test_long_session_survives_many_classification_rows():
    _insert("2026-05-06 09:00:00", "SESSION", "Detection started.")
    conn = database._get_connection()
    conn.executemany(
        "INSERT INTO session_events (timestamp, event_type, detail) VALUES (?, ?, ?)",
        [("2026-05-06 09:30:00", "CLASSIFICATION", "face=f0 risk=REAL trigger=auto")] * 2000,
    )
    conn.commit()
    conn.close()
    _insert("2026-05-06 11:00:00", "SESSION", "Detection stopped.")
    assert [s["started_at"] for s in database.get_all_sessions()] == ["2026-05-06 09:00:00"]


def test_clear_history_removes_everything():
    _insert_two_sessions()
    database.clear_history()
    assert database.get_session_history() == []
    assert database.get_all_sessions() == []


def test_history_table_headers(window):
    headers = [window._history_table.horizontalHeaderItem(i).text() for i in range(4)]
    assert headers == ["Date & time", "Duration", "Alerts", "Result"]


def test_populate_history_table(window):
    window.populate_history_table([
        {"started_at": "2026-05-06 10:00:00", "duration": "00:05:30",
         "alert_count": 2, "result": "Deepfake detected"},
        {"started_at": "2026-05-06 09:00:00", "duration": "00:03:15",
         "alert_count": 0, "result": "Clean"},
    ])
    table = window._history_table
    assert table.rowCount() == 2
    assert [table.item(0, c).text() for c in range(3)] == ["2026-05-06 10:00:00", "00:05:30", "2"]
    assert _badge_text(window, 0) == "Deepfake detected"
    assert _badge_text(window, 1) == "Clean"


def test_refresh_history_loads_sessions_from_db(window):
    _insert_two_sessions()
    window.refresh_history()
    assert not window._history_section.isHidden()
    assert window._history_table.rowCount() == 2
    assert window._history_table.item(0, 0).text() == "2026-05-06 10:00:00"
    assert _badge_text(window, 0) == "Deepfake detected"


def test_refresh_history_hides_section_when_db_empty(window):
    window.refresh_history()
    assert window._history_section.isHidden()
