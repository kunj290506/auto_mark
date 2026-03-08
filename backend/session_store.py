"""
Session Store - Persistent session storage using shelve (Bug 8)
"""

import shelve
import time
from typing import Optional, List, Tuple

SESSION_DB = "sessions_db"


def save_session(session_id: str, data: dict):
    """Save session data to persistent storage"""
    with shelve.open(SESSION_DB) as db:
        db[session_id] = data


def load_session(session_id: str) -> Optional[dict]:
    """Load session data from persistent storage"""
    with shelve.open(SESSION_DB) as db:
        return db.get(session_id)


def delete_session(session_id: str):
    """Delete a session from persistent storage"""
    with shelve.open(SESSION_DB) as db:
        if session_id in db:
            del db[session_id]


def list_all_sessions() -> List[Tuple[str, dict]]:
    """List all sessions with their data"""
    with shelve.open(SESSION_DB) as db:
        return [(sid, db[sid]) for sid in db.keys()]


def session_exists(session_id: str) -> bool:
    """Check if a session exists"""
    with shelve.open(SESSION_DB) as db:
        return session_id in db
