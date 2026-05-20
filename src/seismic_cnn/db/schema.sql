-- SQLite schema for the seismic-cnn project.
-- Run once: sqlite3 project.db < src/seismic_cnn/db/schema.sql

-- Seismic stations in the study area
CREATE TABLE IF NOT EXISTS stations (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    network   TEXT NOT NULL,
    station   TEXT NOT NULL UNIQUE,
    latitude  REAL NOT NULL,
    longitude REAL NOT NULL,
    elevation REAL,
    site_name TEXT
);

-- Merged earthquake catalog (OGS historical + Benz)
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    origin_time TEXT NOT NULL,   -- ISO-8601, tz-naive UTC
    magnitude   REAL,
    source      TEXT NOT NULL,   -- 'historical' | 'benz'
    latitude    REAL,            -- NULL for Benz (no coordinates)
    longitude   REAL
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(origin_time);

-- Per-channel monthly normalisation statistics (cache for Pass 1)
CREATE TABLE IF NOT EXISTS monthly_stats (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    station   TEXT    NOT NULL,
    year      INTEGER NOT NULL,
    month     INTEGER NOT NULL,
    channel   TEXT    NOT NULL,   -- 'HHE' | 'HHN' | 'HHZ'
    mean_val  REAL    NOT NULL,
    peak_val  REAL    NOT NULL,
    UNIQUE(station, year, month, channel)
);

-- Registry of every processed 20-second window
CREATE TABLE IF NOT EXISTS windows (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    station      TEXT    NOT NULL,
    start_time   TEXT    NOT NULL,
    end_time     TEXT    NOT NULL,
    label        TEXT    NOT NULL,   -- 'earthquake' | 'noise'
    split        TEXT    NOT NULL,   -- 'train' | 'test'
    is_augmented INTEGER NOT NULL DEFAULT 0,
    filepath     TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_windows_label ON windows(label, split);
CREATE INDEX IF NOT EXISTS idx_windows_time  ON windows(start_time);

-- Experiment / model run results
CREATE TABLE IF NOT EXISTS model_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name    TEXT    NOT NULL,
    hyperparams   TEXT,            -- JSON string
    val_loss      REAL,
    val_accuracy  REAL,
    val_auc       REAL,
    val_f1        REAL,
    val_precision REAL,
    val_recall    REAL,
    epochs_run    INTEGER,
    trained_at    TEXT NOT NULL    -- ISO-8601 timestamp
);
