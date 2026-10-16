"""
Storage layer for TALAASH - AI Missing Person System.

Two areas:
  - Cases: each missing person is a Case (case number, status,
    last-known location, description, notes over time).
  - Users: login accounts with a role (admin / officer), used for
    TALAASH's single login screen that auto-detects role.

FAISS holds the face embeddings, keyed by the same case id used in
SQLite for cases.
"""
import os
import sqlite3
from datetime import datetime, timedelta

import faiss
import numpy as np

EMBEDDING_DIM = 512
INDEX_PATH = "data/faces_v2.index"
DB_PATH = "data/cases.db"

VALID_STATUSES = ("Active", "Found", "Closed")
VALID_ROLES = ("admin", "officer")


def _ensure_data_dir():
    os.makedirs("data", exist_ok=True)


def init_db():
    _ensure_data_dir()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY,
            case_number TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            contact TEXT,
            last_known_location TEXT,
            description TEXT,
            status TEXT NOT NULL DEFAULT 'Active',
            photo_path TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS case_notes (
            id INTEGER PRIMARY KEY,
            case_id INTEGER NOT NULL,
            note TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES cases (id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            role TEXT NOT NULL DEFAULT 'officer',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY,
            case_id INTEGER NOT NULL,
            camera TEXT,
            confidence REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'Pending',
            created_at TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES cases (id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY,
            case_id INTEGER,
            officer_username TEXT NOT NULL,
            location TEXT,
            description TEXT,
            photo_path TEXT,
            status TEXT NOT NULL DEFAULT 'New',
            created_at TEXT NOT NULL,
            FOREIGN KEY (case_id) REFERENCES cases (id)
        )
        """
    )
    conn.commit()

    _migrate_cases_table(conn)
    _migrate_users_table(conn)
    _migrate_reports_table(conn)
    return conn


# Columns the "cases" table is expected to have, with the SQL type to
# use if a column needs to be added to an older database file.
# CREATE TABLE IF NOT EXISTS only runs on a table that doesn't exist
# yet, so a database created before a new field was added (like
# "description") would otherwise be stuck without it forever. This
# migration adds any missing column on startup, without touching
# existing rows.
_EXPECTED_CASE_COLUMNS = {
    "case_number": "TEXT",
    "name": "TEXT",
    "age": "INTEGER",
    "gender": "TEXT",
    "contact": "TEXT",
    "last_known_location": "TEXT",
    "description": "TEXT",
    "status": "TEXT NOT NULL DEFAULT 'Active'",
    "photo_path": "TEXT",
    "latitude": "REAL",
    "longitude": "REAL",
    "created_at": "TEXT",
}


def _migrate_cases_table(conn):
    existing = {row[1] for row in conn.execute("PRAGMA table_info(cases)").fetchall()}
    for column, col_type in _EXPECTED_CASE_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE cases ADD COLUMN {column} {col_type}")
    conn.commit()


_EXPECTED_REPORT_COLUMNS = {
    "case_id": "INTEGER",
    "officer_username": "TEXT",
    "location": "TEXT",
    "description": "TEXT",
    "photo_path": "TEXT",
    "status": "TEXT NOT NULL DEFAULT 'New'",
    "created_at": "TEXT",
}


def _migrate_reports_table(conn):
    existing = {row[1] for row in conn.execute("PRAGMA table_info(reports)").fetchall()}
    for column, col_type in _EXPECTED_REPORT_COLUMNS.items():
        if column not in existing:
            safe_type = "TEXT" if "NOT NULL" in col_type and "DEFAULT" not in col_type else col_type
            conn.execute(f"ALTER TABLE reports ADD COLUMN {column} {safe_type}")
    conn.commit()


def load_or_create_index():
    _ensure_data_dir()
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    base_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    return faiss.IndexIDMap(base_index)


def save_index(index):
    _ensure_data_dir()
    faiss.write_index(index, INDEX_PATH)


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# ---------------------------------------------------------------- Cases

def _generate_case_number(conn):
    year = datetime.now().year
    count = conn.execute(
        "SELECT COUNT(*) FROM cases WHERE case_number LIKE ?", (f"MP-{year}-%",)
    ).fetchone()[0]
    return f"MP-{year}-{count + 1:04d}"


def add_case(conn, index, name, embedding, age=None, gender=None, contact=None,
             last_known_location=None, description=None, photo_path=None,
             latitude=None, longitude=None):
    case_number = _generate_case_number(conn)
    created_at = datetime.now().isoformat(timespec="seconds")

    cur = conn.execute(
        """
        INSERT INTO cases
            (case_number, name, age, gender, contact, last_known_location,
             description, status, photo_path, latitude, longitude, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'Active', ?, ?, ?, ?)
        """,
        (case_number, name, age, gender, contact, last_known_location,
         description, photo_path, latitude, longitude, created_at),
    )
    case_id = cur.lastrowid
    conn.commit()

    vec = normalize(embedding.astype("float32")).reshape(1, -1)
    ids = np.array([case_id], dtype="int64")
    index.add_with_ids(vec, ids)
    save_index(index)

    return case_id, case_number


def search(index, embedding, top_k=1):
    vec = normalize(embedding.astype("float32")).reshape(1, -1)
    scores, ids = index.search(vec, top_k)
    results = []
    for score, cid in zip(scores[0], ids[0]):
        if cid == -1:
            continue
        results.append((int(cid), float(score)))
    return results


def _row_to_case(row):
    keys = ["id", "case_number", "name", "age", "gender", "contact",
            "last_known_location", "description", "status", "photo_path",
            "latitude", "longitude", "created_at"]
    return dict(zip(keys, row))


CASE_COLUMNS = """id, case_number, name, age, gender, contact,
                  last_known_location, description, status, photo_path,
                  latitude, longitude, created_at"""


def get_case(conn, case_id):
    row = conn.execute(
        f"SELECT {CASE_COLUMNS} FROM cases WHERE id = ?", (case_id,)
    ).fetchone()
    return _row_to_case(row) if row else None


def list_cases(conn, status=None):
    if status:
        rows = conn.execute(
            f"SELECT {CASE_COLUMNS} FROM cases WHERE status = ? ORDER BY id DESC", (status,)
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {CASE_COLUMNS} FROM cases ORDER BY id DESC"
        ).fetchall()
    return [_row_to_case(row) for row in rows]


def update_case_status(conn, case_id, status):
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {VALID_STATUSES}")
    conn.execute("UPDATE cases SET status = ? WHERE id = ?", (status, case_id))
    conn.commit()


def add_case_note(conn, case_id, note):
    created_at = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO case_notes (case_id, note, created_at) VALUES (?, ?, ?)",
        (case_id, note, created_at),
    )
    conn.commit()


def list_case_notes(conn, case_id):
    rows = conn.execute(
        "SELECT id, note, created_at FROM case_notes WHERE case_id = ? ORDER BY id DESC",
        (case_id,),
    ).fetchall()
    return [{"id": r[0], "note": r[1], "created_at": r[2]} for r in rows]


def list_cases_missing_coordinates(conn):
    """Cases with a location on file but no coordinates yet -- either
    created before geocoding was set up, or geocoding failed at the
    time (no API key configured yet, address not found, etc)."""
    rows = conn.execute(
        f"""SELECT {CASE_COLUMNS} FROM cases
            WHERE last_known_location IS NOT NULL
              AND last_known_location != ''
              AND (latitude IS NULL OR longitude IS NULL)"""
    ).fetchall()
    return [_row_to_case(row) for row in rows]


def update_case_coordinates(conn, case_id, latitude, longitude):
    conn.execute(
        "UPDATE cases SET latitude = ?, longitude = ? WHERE id = ?",
        (latitude, longitude, case_id),
    )
    conn.commit()


def case_stats(conn):
    """Powers the Home dashboard's summary cards."""
    total = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    active = conn.execute("SELECT COUNT(*) FROM cases WHERE status = 'Active'").fetchone()[0]
    found = conn.execute("SELECT COUNT(*) FROM cases WHERE status = 'Found'").fetchone()[0]
    today = datetime.now().date().isoformat()
    matches_today = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE created_at LIKE ?", (f"{today}%",)
    ).fetchone()[0]
    return {
        "total_missing": total,
        "active_cases": active,
        "persons_found": found,
        "matches_today": matches_today,
    }


def _daily_counts(conn, table, days):
    """Per-day row counts for the last `days` days (today included),
    with zeros filled in for days that had nothing, so charts don't
    have gaps."""
    start = datetime.now().date() - timedelta(days=days - 1)
    rows = conn.execute(
        f"SELECT substr(created_at, 1, 10), COUNT(*) FROM {table} "
        "WHERE created_at >= ? GROUP BY 1",
        (start.isoformat(),),
    ).fetchall()
    counts = dict(rows)
    series = []
    for i in range(days):
        day = (start + timedelta(days=i)).isoformat()
        series.append({"date": day, "count": counts.get(day, 0)})
    return series


def analytics_stats(conn, days=7):
    """Everything the Analytics dashboard needs, in one query pass."""
    cases_by_status = {s: 0 for s in VALID_STATUSES}
    for status, n in conn.execute("SELECT status, COUNT(*) FROM cases GROUP BY status"):
        cases_by_status[status] = n
    total_cases = sum(cases_by_status.values())

    detections_by_status = {"Pending": 0, "Verified": 0, "Rejected": 0}
    for status, n in conn.execute("SELECT status, COUNT(*) FROM detections GROUP BY status"):
        detections_by_status[status] = n
    total_detections = sum(detections_by_status.values())

    reports_by_status = {"New": 0, "Reviewed": 0}
    for status, n in conn.execute("SELECT status, COUNT(*) FROM reports GROUP BY status"):
        reports_by_status[status] = n
    total_reports = sum(reports_by_status.values())

    # Confidence is stored as 0-1; anything below the match threshold
    # shouldn't exist but would land in the lowest bucket if it did.
    buckets = {"55-65%": 0, "65-75%": 0, "75-85%": 0, "85-100%": 0}
    confidences = [r[0] for r in conn.execute("SELECT confidence FROM detections")]
    for c in confidences:
        if c < 0.65:
            buckets["55-65%"] += 1
        elif c < 0.75:
            buckets["65-75%"] += 1
        elif c < 0.85:
            buckets["75-85%"] += 1
        else:
            buckets["85-100%"] += 1
    avg_confidence = (sum(confidences) / len(confidences)) if confidences else None

    cameras = [
        {"camera": cam or "Unknown", "count": n}
        for cam, n in conn.execute(
            "SELECT camera, COUNT(*) FROM detections GROUP BY camera ORDER BY COUNT(*) DESC LIMIT 5"
        )
    ]

    reviewed = detections_by_status["Verified"] + detections_by_status["Rejected"]
    return {
        "days": days,
        "totals": {
            "cases": total_cases,
            "detections": total_detections,
            "reports": total_reports,
        },
        # Share of cases marked Found -- the headline outcome metric.
        "resolution_rate": (cases_by_status["Found"] / total_cases) if total_cases else None,
        # Of the alerts an officer actually reviewed, how many were
        # confirmed as real. A rough, honest read on alert quality.
        "confirmation_rate": (detections_by_status["Verified"] / reviewed) if reviewed else None,
        "avg_confidence": avg_confidence,
        "cases_by_status": cases_by_status,
        "detections_by_status": detections_by_status,
        "reports_by_status": reports_by_status,
        "confidence_buckets": buckets,
        "cameras": cameras,
        "detections_per_day": _daily_counts(conn, "detections", days),
        "reports_per_day": _daily_counts(conn, "reports", days),
    }


VALID_DETECTION_STATUSES = ("Pending", "Verified", "Rejected")


def add_detection(conn, case_id, camera, confidence):
    created_at = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO detections (case_id, camera, confidence, status, created_at) VALUES (?, ?, ?, 'Pending', ?)",
        (case_id, camera, confidence, created_at),
    )
    conn.commit()
    return cur.lastrowid


def list_detections(conn, period=None, limit=None):
    """period: None (all), 'today', 'week', or 'month'."""
    query = """
        SELECT d.id, d.case_id, c.case_number, c.name, d.camera,
               d.confidence, d.status, d.created_at
        FROM detections d
        JOIN cases c ON c.id = d.case_id
    """
    params = []

    if period == "today":
        query += " WHERE d.created_at >= ?"
        params.append(datetime.now().date().isoformat())
    elif period == "week":
        query += " WHERE d.created_at >= ?"
        params.append((datetime.now() - timedelta(days=7)).isoformat())
    elif period == "month":
        query += " WHERE d.created_at >= ?"
        params.append((datetime.now() - timedelta(days=30)).isoformat())

    query += " ORDER BY d.id DESC"
    if limit:
        query += f" LIMIT {int(limit)}"

    rows = conn.execute(query, params).fetchall()
    keys = ["id", "case_id", "case_number", "name", "camera", "confidence", "status", "created_at"]
    return [dict(zip(keys, row)) for row in rows]


def update_detection_status(conn, detection_id, status):
    if status not in VALID_DETECTION_STATUSES:
        raise ValueError(f"status must be one of {VALID_DETECTION_STATUSES}")
    conn.execute("UPDATE detections SET status = ? WHERE id = ?", (status, detection_id))
    conn.commit()


# ------------------------------------------------------------- Reports

VALID_REPORT_STATUSES = ("New", "Reviewed")


def add_report(conn, case_id, officer_username, location=None, description=None, photo_path=None):
    created_at = datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        """
        INSERT INTO reports (case_id, officer_username, location, description, photo_path, status, created_at)
        VALUES (?, ?, ?, ?, ?, 'New', ?)
        """,
        (case_id, officer_username, location, description, photo_path, created_at),
    )
    conn.commit()
    return cur.lastrowid


def list_reports(conn, officer_username=None):
    query = """
        SELECT r.id, r.case_id, c.case_number, c.name, r.officer_username,
               r.location, r.description, r.photo_path, r.status, r.created_at
        FROM reports r
        LEFT JOIN cases c ON c.id = r.case_id
    """
    params = []
    if officer_username:
        query += " WHERE r.officer_username = ?"
        params.append(officer_username)
    query += " ORDER BY r.id DESC"

    rows = conn.execute(query, params).fetchall()
    keys = ["id", "case_id", "case_number", "name", "officer_username",
            "location", "description", "photo_path", "status", "created_at"]
    return [dict(zip(keys, row)) for row in rows]


def update_report_status(conn, report_id, status):
    if status not in VALID_REPORT_STATUSES:
        raise ValueError(f"status must be one of {VALID_REPORT_STATUSES}")
    conn.execute("UPDATE reports SET status = ? WHERE id = ?", (status, report_id))
    conn.commit()


# ---------------------------------------------------------------- Users

_EXPECTED_USER_COLUMNS = {
    "username": "TEXT UNIQUE NOT NULL",
    "password_hash": "TEXT NOT NULL",
    "full_name": "TEXT",
    "email": "TEXT",
    "role": "TEXT NOT NULL DEFAULT 'officer'",
    "created_at": "TEXT",
}


def _migrate_users_table(conn):
    existing = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    for column, col_type in _EXPECTED_USER_COLUMNS.items():
        if column not in existing:
            # SQLite won't let ALTER TABLE add a column with a
            # UNIQUE or NOT NULL constraint that has no default, so
            # any such columns are added in a relaxed form here --
            # they're only enforced strictly when the table is
            # created fresh, via the CREATE TABLE statement above.
            safe_type = "TEXT" if ("UNIQUE" in col_type or "NOT NULL" in col_type) else col_type
            conn.execute(f"ALTER TABLE users ADD COLUMN {column} {safe_type}")
    conn.commit()


def create_user(conn, username, password_hash, role="officer", full_name=None, email=None):
    if role not in VALID_ROLES:
        raise ValueError(f"role must be one of {VALID_ROLES}")
    created_at = datetime.now().isoformat(timespec="seconds")
    conn.execute(
        "INSERT INTO users (username, password_hash, full_name, email, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (username, password_hash, full_name, email, role, created_at),
    )
    conn.commit()


def get_user_by_username(conn, username):
    row = conn.execute(
        "SELECT id, username, password_hash, full_name, email, role FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    if row is None:
        return None
    keys = ["id", "username", "password_hash", "full_name", "email", "role"]
    return dict(zip(keys, row))


def update_user_password(conn, username, password_hash):
    conn.execute("UPDATE users SET password_hash = ? WHERE username = ?", (password_hash, username))
    conn.commit()


def update_user_profile(conn, username, full_name=None, email=None):
    """Updates only the fields actually passed in (None means 'leave
    this field alone', not 'clear it')."""
    fields = []
    values = []
    if full_name is not None:
        fields.append("full_name = ?")
        values.append(full_name)
    if email is not None:
        fields.append("email = ?")
        values.append(email)

    if not fields:
        return

    values.append(username)
    conn.execute(f"UPDATE users SET {', '.join(fields)} WHERE username = ?", values)
    conn.commit()


def update_user_email(conn, username, email):
    """Kept for the one-off set_user_email.py script; new code
    should use update_user_profile instead."""
    update_user_profile(conn, username, email=email)


def delete_user(conn, username):
    conn.execute("DELETE FROM users WHERE username = ?", (username,))
    conn.commit()


def list_users(conn):
    rows = conn.execute(
        "SELECT id, username, full_name, email, role, created_at FROM users ORDER BY id"
    ).fetchall()
    keys = ["id", "username", "full_name", "email", "role", "created_at"]
    return [dict(zip(keys, r)) for r in rows]