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


def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute(CREATE_TABLE_SQL)
    conn.commit()
    return conn


def log_event(
    event_type: str,
    detail: str = "",
    peak_confidence: float = None,
    breach_duration_sec: float = None,
    faces_detected: int = None,
    api_response_ms: float = None,
    api_request_id: str = None,
) -> None:
    """
    Insert a session event record into the local SQLite database.

    Args:
        event_type:          Category tag (e.g. 'ALERT', 'DISMISS', 'SESSION', 'ERROR').
        detail:              Human-readable description of the event.
        peak_confidence:     Peak P_fake score at time of event (optional).
        breach_duration_sec: Duration of alert breach in seconds (optional).
        faces_detected:      Number of faces detected at time of event (optional).
        api_response_ms:     BitMind API round-trip response time in ms (optional).
        api_request_id:      BitMind API request identifier (optional).
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
        conn.execute(sql, (
            timestamp, event_type, detail,
            peak_confidence, breach_duration_sec, faces_detected,
            api_response_ms, api_request_id
        ))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"[WARNING] Database write failed: {e}")


def get_session_history(limit: int = 100) -> list:
    """
    Retrieve the most recent session events.

    Args:
        limit: Maximum number of records to return (default: 100).

    Returns:
        List of dicts with keys: id, timestamp, event_type, detail,
        peak_confidence, breach_duration_sec, faces_detected, api_response_ms, api_request_id.
    """
    sql = """
        SELECT id, timestamp, event_type, detail,
               peak_confidence, breach_duration_sec, faces_detected,
               api_response_ms, api_request_id
        FROM session_events
        ORDER BY id DESC
        LIMIT ?
    """
    try:
        conn = _get_connection()
        cursor = conn.execute(sql, (limit,))
        rows = cursor.fetchall()
        conn.close()
        keys = ["id", "timestamp", "event_type", "detail",
                "peak_confidence", "breach_duration_sec", "faces_detected",
                "api_response_ms", "api_request_id"]
        return [dict(zip(keys, row)) for row in rows]
    except sqlite3.Error as e:
        print(f"[WARNING] Database read failed: {e}")
        return []


def clear_history() -> None:
    """Delete all session event records. Used for testing only."""
    try:
        conn = _get_connection()
        conn.execute("DELETE FROM session_events")
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"[WARNING] Database clear failed: {e}")


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
    sql = """
        SELECT id, timestamp, event_type, detail
        FROM session_events
        ORDER BY id DESC
        LIMIT ?
    """
    try:
        conn = _get_connection()
        cursor = conn.execute(sql, (limit * 10,))  # fetch more to reconstruct sessions
        rows = cursor.fetchall()
        conn.close()
        
        # Reconstruct sessions from event log
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
                    if evt_type == "SESSION" and evt_detail and "Detection stopped" in evt_detail:
                        stop_idx = j
                        break
                
                if stop_idx is not None:
                    ended_at = rows[stop_idx][1]
                    
                    # Count ALERT events between start and stop
                    alert_count = 0
                    for k in range(start_idx, stop_idx):
                        if rows[k][2] == "ALERT":
                            alert_count += 1
                    
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
                    
                    result_text = "Deepfake detected" if alert_count > 0 else "Clean"
                    
                    sessions.append({
                        "started_at": started_at,
                        "duration": duration_str,
                        "alert_count": alert_count,
                        "result": result_text
                    })
                    
                    i = stop_idx + 1
                else:
                    i += 1
            else:
                i += 1
        
        # Return most recent sessions first
        return list(reversed(sessions))[:limit]
        
    except sqlite3.Error as e:
        print(f"[WARNING] Database read failed: {e}")
        return []
