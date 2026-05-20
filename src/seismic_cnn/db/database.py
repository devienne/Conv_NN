"""SQLite database connection and helper functions."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from seismic_cnn.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "data" / "project.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_connection(db_path=None) -> sqlite3.Connection:
    """Return a sqlite3 connection with row_factory set to Row."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    """Create all tables from schema.sql (idempotent)."""
    conn = get_connection(db_path)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def insert_events(events: list[dict], db_path=None):
    """Bulk-insert earthquake events. Each dict must have: origin_time, source.

    Optional keys: magnitude, latitude, longitude.
    """
    conn = get_connection(db_path)
    conn.executemany(
        "INSERT OR IGNORE INTO events(origin_time, magnitude, source, latitude, longitude) "
        "VALUES(:origin_time, :magnitude, :source, :latitude, :longitude)",
        events,
    )
    conn.commit()
    conn.close()


def query_events(start: str, end: str, db_path=None) -> list:
    """Return events with origin_time in [start, end] (ISO strings)."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM events WHERE origin_time >= ? AND origin_time <= ? "
        "ORDER BY origin_time",
        (start, end),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Monthly stats cache
# ---------------------------------------------------------------------------

def upsert_monthly_stats(station: str, year: int, month: int,
                          channel: str, mean_val: float, peak_val: float,
                          db_path=None):
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO monthly_stats(station, year, month, channel, mean_val, peak_val) "
        "VALUES(?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(station, year, month, channel) DO UPDATE SET "
        "mean_val=excluded.mean_val, peak_val=excluded.peak_val",
        (station, year, month, channel, mean_val, peak_val),
    )
    conn.commit()
    conn.close()


def get_monthly_stats(station: str, year: int, month: int,
                       db_path=None) -> dict:
    """Return {channel: {mean, peak}} or {} if not cached."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT channel, mean_val, peak_val FROM monthly_stats "
        "WHERE station=? AND year=? AND month=?",
        (station, year, month),
    ).fetchall()
    conn.close()
    return {r["channel"]: {"mean": r["mean_val"], "peak": r["peak_val"]} for r in rows}


# ---------------------------------------------------------------------------
# Windows registry
# ---------------------------------------------------------------------------

def insert_windows(windows: list[dict], db_path=None):
    """Bulk-insert window records. Each dict must have:
    station, start_time, end_time, label, split, is_augmented, filepath.
    Uses INSERT OR IGNORE so re-running the pipeline never duplicates rows.
    """
    if not windows:
        return
    conn = get_connection(db_path)
    conn.executemany(
        "INSERT OR IGNORE INTO windows"
        "(station, start_time, end_time, label, split, is_augmented, filepath) "
        "VALUES(:station, :start_time, :end_time, :label, :split, :is_augmented, :filepath)",
        windows,
    )
    conn.commit()
    conn.close()


def count_windows(db_path=None) -> dict:
    """Return {(split, label): count} from the windows table."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT split, label, COUNT(*) as n FROM windows GROUP BY split, label"
    ).fetchall()
    conn.close()
    return {(r["split"], r["label"]): r["n"] for r in rows}


def get_windows_summary(db_path=None) -> list[dict]:
    """Return window counts grouped by split, label, and is_augmented."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT split, label, is_augmented, COUNT(*) as count "
        "FROM windows GROUP BY split, label, is_augmented "
        "ORDER BY split, label, is_augmented"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_windows_monthly(db_path=None) -> list[dict]:
    """Return window counts per calendar month per label, across all splits."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT strftime('%Y-%m', start_time) as month, label, COUNT(*) as count "
        "FROM windows GROUP BY month, label ORDER BY month, label"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Model runs
# ---------------------------------------------------------------------------

def log_model_run(model_name: str, hyperparams: dict, metrics: dict,
                   epochs_run: int, db_path=None):
    """Insert a training result into model_runs."""
    conn = get_connection(db_path)
    conn.execute(
        "INSERT INTO model_runs"
        "(model_name, hyperparams, val_loss, val_accuracy, val_auc, "
        " val_f1, val_precision, val_recall, epochs_run, trained_at) "
        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            model_name,
            json.dumps(hyperparams),
            metrics.get("loss"),
            metrics.get("accuracy"),
            metrics.get("auc"),
            metrics.get("f1"),
            metrics.get("precision"),
            metrics.get("recall"),
            epochs_run,
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def list_model_runs(db_path=None) -> list:
    """Return all model runs ordered by val_auc descending."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT * FROM model_runs ORDER BY val_auc DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
