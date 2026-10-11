"""
database.py

Local SQLite backend for session event logging.
All data is stored in a single file on the host machine.
No data is transmitted to external servers.
"""

import sqlite3
import os
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS session_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp           TEXT    NOT NULL,
    event_type          TEXT    NOT NULL,
    detail              TEXT,
    peak_confidence     REAL,
    breach_duration_sec REAL,
    faces_detected      INTEGER,
    api_response_ms     REAL,
    api_request_id      TEXT
);
"""


SESSION_EXPORT_FIELDS = ["timestamp", "event_type", "detail", "peak_confidence", "api_response_ms"]


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(CREATE_TABLE_SQL)
    return conn


def is_available() -> bool:
    """Whether the session database can be opened."""
    try:
        _get_connection().close()
        return True
    except sqlite3.Error:
        return False


def log_event(
    event_type: str,
    detail: str = "",
    peak_confidence: float = None,
    breach_duration_sec: float = None,
    faces_detected: int = None,
    api_response_ms: float = None,
    api_request_id: str = None,
) -> int | None:
    """
    Insert a session event record into the local SQLite database.

    Args:
        event_type:          Category tag (e.g. 'ALERT', 'DISMISS', 'SESSION', 'ERROR', 'CLASSIFICATION').
        detail:              Human-readable description of the event.
        peak_confidence:     Peak P_fake score at time of event (optional).
        breach_duration_sec: Duration of alert breach in seconds (optional).
        faces_detected:      Number of faces detected at time of event (optional).
        api_response_ms:     BitMind API round-trip response time in ms (optional).
        api_request_id:      BitMind API request identifier (optional).

    Returns:
        The new row's id, or None if the write failed.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sql = """
        INSERT INTO session_events
            (timestamp, event_type, detail, peak_confidence,
             breach_duration_sec, faces_detected, api_response_ms, api_request_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    try:
        conn = _get_connection()
        cursor = conn.execute(sql, (
            timestamp, event_type, detail,
            peak_confidence, breach_duration_sec, faces_detected,
            api_response_ms, api_request_id
        ))
        conn.commit()
        conn.close()
        return cursor.lastrowid
    except sqlite3.Error as e:
        print(f"[WARNING] Database write failed: {e}")
        return None


def _find_session_id_range(conn: sqlite3.Connection, started_at: str) -> tuple | None:
    """Return (start_id, stop_id_or_None) for the session that began at started_at."""
    cursor = conn.execute(
        "SELECT id FROM session_events WHERE event_type='SESSION' AND detail LIKE '%Detection started%' AND timestamp=? ORDER BY id ASC LIMIT 1",
        (started_at,)
    )
    row = cursor.fetchone()
    if row is None:
        return None
    start_id = row[0]
    cursor = conn.execute(
        "SELECT id, detail FROM session_events WHERE event_type='SESSION' AND id > ? ORDER BY id ASC LIMIT 1",
        (start_id,)
    )
    row = cursor.fetchone()
    if row is None:
        return start_id, None
    next_id, next_detail = row
    # A start followed directly by another start was never stopped (the app
    # crashed, or it predates Discard deleting its events) — it ends just
    # before the next session rather than claiming that session's stop.
    return start_id, (next_id if "Detection stopped" in next_detail else next_id - 1)


def delete_session(started_at: str) -> None:
    """Delete all events belonging to the session that started at the given timestamp."""
    try:
        conn = _get_connection()
        id_range = _find_session_id_range(conn, started_at)
        if id_range is None:
            conn.close()
            return
        start_id, stop_id = id_range
        if stop_id is not None:
            conn.execute("DELETE FROM session_events WHERE id BETWEEN ? AND ?", (start_id, stop_id))
        else:
            conn.execute("DELETE FROM session_events WHERE id >= ?", (start_id,))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"[WARNING] Database delete failed: {e}")


def get_session_events(started_at: str) -> list:
    """Return the events of the session that started at started_at, oldest first, keyed by SESSION_EXPORT_FIELDS."""
    try:
        conn = _get_connection()
        id_range = _find_session_id_range(conn, started_at)
        if id_range is None:
            conn.close()
            return []
        start_id, stop_id = id_range
        where, params = ("id BETWEEN ? AND ?", id_range) if stop_id is not None else ("id >= ?", (start_id,))
        rows = conn.execute(
            f"SELECT {', '.join(SESSION_EXPORT_FIELDS)} FROM session_events WHERE {where} ORDER BY id", params
        ).fetchall()
        conn.close()
        return [dict(zip(SESSION_EXPORT_FIELDS, row)) for row in rows]
    except sqlite3.Error as e:
        print(f"[WARNING] Database read failed: {e}")
        return []


def delete_events_from(event_id: int) -> None:
    """Delete the event with this id and every later one — used to discard the session it started."""
    try:
        conn = _get_connection()
        conn.execute("DELETE FROM session_events WHERE id >= ?", (event_id,))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"[WARNING] Database delete failed: {e}")


def clear_history() -> None:
    """Delete all session event records."""
    try:
        conn = _get_connection()
        conn.execute("DELETE FROM session_events")
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"[WARNING] Database clear failed: {e}")


def _reconstruct_sessions(rows: list) -> list:
    """
    Pair SESSION start/stop rows (oldest-first: id ASC) into completed
    sessions with duration and alert count. Sessions with no matching stop
    event are skipped (still in progress, truncated, or followed by another
    start before any stop).

    Args:
        rows: list of (id, timestamp, event_type, detail) tuples, oldest-first.

    Returns:
        List of dicts with keys: started_at, duration, alert_count, result,
        in the same (oldest-first) order as the input rows.
    """
    sessions = []
    i = 0
    while i < len(rows):
        row_id, timestamp, event_type, detail = rows[i]

        # Look for a session start event
        if event_type == "SESSION" and detail and "Detection started" in detail:
            started_at = timestamp
            start_idx = i

            # Find the corresponding stop event
            stop_idx = None
            for j in range(i + 1, len(rows)):
                evt_id, ts, evt_type, evt_detail = rows[j]
                if evt_type == "SESSION" and evt_detail and "Detection started" in evt_detail:
                    break  # never stopped — must not claim the next session's stop
                if evt_type == "SESSION" and evt_detail and "Detection stopped" in evt_detail:
                    stop_idx = j
                    break

            if stop_idx is not None:
                ended_at = rows[stop_idx][1]

                # Count ALERT events between start and stop
                alert_count = sum(1 for k in range(start_idx, stop_idx) if rows[k][2] == "ALERT")

                # Compute duration
                try:
                    fmt = "%Y-%m-%d %H:%M:%S"
                    start_dt = datetime.strptime(started_at, fmt)
                    end_dt = datetime.strptime(ended_at, fmt)
                    delta = end_dt - start_dt
                    total_seconds = int(delta.total_seconds())
                    h, rem = divmod(total_seconds, 3600)
                    m, s = divmod(rem, 60)
                    duration_str = f"{h:02d}:{m:02d}:{s:02d}"
                except Exception:
                    duration_str = "—"

                sessions.append({
                    "started_at": started_at,
                    "duration": duration_str,
                    "alert_count": alert_count,
                    "result": "Deepfake detected" if alert_count > 0 else "Clean",
                })

                i = stop_idx + 1
            else:
                i += 1
        else:
            i += 1

    return sessions


def get_all_sessions(limit: int = 100) -> list:
    """
    Retrieve all completed sessions reconstructed from event log.

    A session is defined as:
    - Starts with event_type='SESSION' and detail containing 'Detection started'
    - Ends with event_type='SESSION' and detail containing 'Detection stopped'
    - Alert count is the number of ALERT events between start and stop

    Args:
        limit: Maximum number of sessions to return (default: 100).

    Returns:
        List of dicts with keys: started_at, duration, alert_count, result.
        Sorted by most recent first.
    """
    # Only SESSION/ALERT rows matter for reconstruction; without this filter a
    # long session's CLASSIFICATION rows would push its start row out of the
    # LIMIT window and the session would vanish from history.
    sql = """
        SELECT id, timestamp, event_type, detail
        FROM session_events
        WHERE event_type IN ('SESSION', 'ALERT')
        ORDER BY id DESC
        LIMIT ?
    """
    try:
        conn = _get_connection()
        cursor = conn.execute(sql, (limit * 10,))  # fetch more to reconstruct sessions
        rows = cursor.fetchall()
        conn.close()

        # The query above orders newest-first, but pairing a "started" event
        # with its "stopped" event requires scanning forward in time (a stop
        # always has a higher id than its start). Reverse to chronological
        # (oldest-first) order before pairing, then reverse the reconstructed
        # sessions back to newest-first for display at the end.
        rows = list(reversed(rows))
        sessions = _reconstruct_sessions(rows)

        # Return most recent sessions first
        return list(reversed(sessions))[:limit]

    except sqlite3.Error as e:
        print(f"[WARNING] Database read failed: {e}")
        return []
