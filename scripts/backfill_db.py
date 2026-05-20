"""One-time database backfill.

Reads existing catalog CSVs and window files from disk and inserts them into
the SQLite database. Safe to run multiple times — INSERT OR IGNORE prevents
duplicates.

Usage
-----
python scripts/backfill_db.py
python scripts/backfill_db.py --config configs/experiment_convnetquake.yaml
"""

import argparse
import os
import pandas as pd

from seismic_cnn.config import load_config
from seismic_cnn.db.database import init_db, insert_events, insert_windows


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def backfill_events(cfg):
    historical_csv = os.path.join(cfg.raw_catalogs_dir, "historical.csv")
    if os.path.exists(historical_csv):
        df = pd.read_csv(historical_csv)
        records = [
            {
                "origin_time": str(r.get("origintime", ""))[:19],
                "magnitude":   r.get("magnitude"),
                "source":      "historical",
                "latitude":    r.get("latitude"),
                "longitude":   r.get("longitude"),
            }
            for r in df.to_dict("records")
        ]
        insert_events(records)
        print(f"  historical.csv  → {len(records)} events inserted (or already present)")
    else:
        print(f"  WARNING: {historical_csv} not found — skipping historical events")

    benz_csv = os.path.join(cfg.raw_catalogs_dir, "Benz_catalog.csv")
    if os.path.exists(benz_csv):
        df = pd.read_csv(benz_csv)
        records = [
            {
                "origin_time": str(r.get("origintime", ""))[:19],
                "magnitude":   r.get("Magnitude"),
                "source":      "benz",
                "latitude":    None,
                "longitude":   None,
            }
            for r in df.to_dict("records")
        ]
        insert_events(records)
        print(f"  Benz_catalog.csv → {len(records)} events inserted (or already present)")
    else:
        print(f"  INFO: {benz_csv} not found — skipping Benz events")


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def _parse_filename(fname: str):
    """Parse '2014-02-15_00-19-00_2014-02-15_00-19-20[...].csv' → (start, end) strings."""
    stem  = fname.replace(".csv", "")
    parts = stem.split("_")
    # parts[0]='2014-02-15', parts[1]='00-19-00', parts[2]='2014-02-15', parts[3]='00-19-20'
    start = f"{parts[0]} {parts[1].replace('-', ':', 2)}"
    end   = f"{parts[2]} {parts[3].replace('-', ':', 2)}"
    return start, end


def backfill_windows(cfg):
    dirs = {
        ("train", "earthquake"): os.path.join(cfg.windows_dir, "train", "earthquake"),
        ("train", "noise"):      os.path.join(cfg.windows_dir, "train", "noise"),
        ("test",  "earthquake"): os.path.join(cfg.windows_dir, "test",  "earthquake"),
        ("test",  "noise"):      os.path.join(cfg.windows_dir, "test",  "noise"),
    }
    total = 0
    for (split, label), directory in dirs.items():
        if not os.path.isdir(directory):
            print(f"  WARNING: {directory} does not exist — skipping")
            continue

        files   = [f for f in os.listdir(directory) if f.endswith(".csv")]
        records = []
        skipped = 0
        for fname in files:
            try:
                start, end = _parse_filename(fname)
                records.append({
                    "station":      cfg.station,
                    "start_time":   start,
                    "end_time":     end,
                    "label":        label,
                    "split":        split,
                    "is_augmented": 1 if "_aug" in fname else 0,
                    "filepath":     os.path.join(directory, fname),
                })
            except Exception:
                skipped += 1

        insert_windows(records)
        msg = f"  {split}/{label}: {len(records)} windows registered"
        if skipped:
            msg += f"  ({skipped} filenames could not be parsed — skipped)"
        print(msg)
        total += len(records)

    print(f"\n  Total windows registered: {total}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg):
    init_db()

    print("=== Backfilling events ===")
    backfill_events(cfg)

    print("\n=== Backfilling windows ===")
    backfill_windows(cfg)

    print("\nBackfill complete. Re-run as many times as needed — duplicates are ignored.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate SQLite from existing files.")
    parser.add_argument("--config", default=None, help="Path to experiment YAML")
    parser.add_argument("--local",  default=None, help="Path to local YAML")
    args = parser.parse_args()
    main(load_config(args.config, args.local))
