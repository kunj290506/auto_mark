"""Session Store - simple JSON-backed persistent session storage."""

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SESSION_DB = Path("sessions_db.json")
_LOCK = threading.Lock()


def _read_all_sessions() -> Dict[str, dict]:
    if not SESSION_DB.exists():
        return {}
    try:
        data = json.loads(SESSION_DB.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_all_sessions(data: Dict[str, dict]):
    tmp_path = SESSION_DB.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(SESSION_DB)


def save_session(session_id: str, data: dict):
    """Save session data to persistent storage."""
    with _LOCK:
        sessions = _read_all_sessions()
        sessions[session_id] = data
        _write_all_sessions(sessions)


def load_session(session_id: str) -> Optional[dict]:
    """Load session data from persistent storage."""
    with _LOCK:
        return _read_all_sessions().get(session_id)


def delete_session(session_id: str):
    """Delete a session from persistent storage."""
    with _LOCK:
        sessions = _read_all_sessions()
        if session_id in sessions:
            del sessions[session_id]
            _write_all_sessions(sessions)


def list_all_sessions() -> List[Tuple[str, dict]]:
    """List all sessions with their data."""
    with _LOCK:
        sessions = _read_all_sessions()
        return [(sid, payload) for sid, payload in sessions.items()]


def session_exists(session_id: str) -> bool:
    """Check if a session exists."""
    with _LOCK:
        return session_id in _read_all_sessions()
